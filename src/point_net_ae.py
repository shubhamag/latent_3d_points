'''
Created on January 26, 2017

@author: optas
'''
import pdb
import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from . in_out import create_dir
from . autoencoder import AutoEncoder
from . general_utils import apply_augmentations

try:    
    from .. external.structural_losses.tf_nndistance import nn_distance
    from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
    

class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, graph, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss(self):

        c = self.configuration
        disc_kwargs= {}
        with tf.variable_scope("discriminator") as scope:
            _, self.disc_z = self.discriminator(self.z, scope = scope, **disc_kwargs)
            self.noise = tf.random_normal([c.batch_size,self.z.get_shape().as_list()[1]]) / 10

            _, self.disc_n = self.discriminator(self.noise, reuse=True, scope = scope, **disc_kwargs)
        self.loss_d = tf.reduce_mean(self.disc_n) - tf.reduce_mean(self.disc_z)
        self.loss_g = tf.reduce_mean(self.disc_z)
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = self.noise*epsilon + (1-epsilon)*self.z
        with tf.variable_scope('discriminator') as scope:
            self.d_hat_prob, self.d_hat = self.discriminator(x_hat, reuse=True, scope=scope)
        gradients = tf.gradients(self.d_hat, x_hat)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
        self.loss_d += gradient_penalty

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
        if c.adv_ae:
            self.loss += (self.loss_g/100)
        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def discriminator(self, data, reuse=None, scope='disc'):
        with tf.variable_scope(scope, reuse=reuse):
            layer = tf.contrib.layers.fully_connected(data, 256)
            # layer = tf.contrib.layers.fully_connected(layer, 512)
            layer = tf.contrib.layers.fully_connected(layer, 128)
            layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
            prob = tf.nn.sigmoid(layer)
        return prob, layer

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_vars = tf.trainable_variables()
        d_params = [v for v in train_vars if '/discriminator/' in v.name ]
        g_params = [v for v in train_vars if '/discriminator/' not in v.name]

        self.train_step = self.optimizer.minimize(self.loss, var_list=g_params)

        self.d_step = self.optimizer.minimize(self.loss_d/10, var_list=d_params)

    def _single_epoch_train(self, train_data, configuration, only_fw=False,mask_type=0):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        epoch_loss_d=0
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            if(mask_type==0):
                fit = self.partial_fit_without_mask
                print("using partial_fit_without_mask")
            else:
                import functools
                fit = functools.partial(self.partial_fit,mask_type=mask_type)
                print("training partial_fit with mask_type " + str(mask_type))

        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit(batch_i, original_data)
            else:
                _, loss,loss_d = fit(batch_i)

            # Compute average loss
            epoch_loss += loss
            epoch_loss_d+=loss_d
        epoch_loss /= n_batches
        epoch_loss_d /=n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        
        return epoch_loss, duration,epoch_loss_d
    #
    # def _single_epoch_train_global(self, train_data, configuration, only_fw=False):
    #     n_examples = train_data.num_examples
    #     epoch_loss = 0.
    #     batch_size = configuration.batch_size
    #     n_batches = int(n_examples / batch_size)
    #     start_time = time.time()
    #
    #     if only_fw:
    #         fit = self.reconstruct
    #     else:
    #         fit = self.partial_fit
    #
    #     # Loop over all batches
    #     for _ in xrange(n_batches):
    #         rand_points = np.random.choice(train_data.shape[1], 100)
    #         if self.is_denoising:
    #             original_data, _, batch_i = train_data.next_batch(batch_size)
    #             if batch_i is None:  # In this case the denoising concern only the augmentation.
    #                 batch_i = original_data
    #         else:
    #             batch_i, _, _ = train_data.next_batch(batch_size)
    #
    #
    #
    #         batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.
    #         batch_temp = np.transpose(np.expand_dims(batch_i[:,rand_points,:], 3),[0,3,2,1])
    #         batch_diff = np.sum(np.square(np.expand_dims(batch_i,3) - batch_temp), axis=2)
    #         batch_ext = np.concatenate((batch_i, batch_diff), axis = 2)
    #         pdb.set_trace()
    #
    #         if self.is_denoising:
    #             _, loss = fit(batch_i, original_data)
    #         else:
    #             _, loss = fit(batch_ext, batch_i)
    #
    #         # Compute average loss
    #         epoch_loss += loss
    #     epoch_loss /= n_batches
    #     duration = time.time() - start_time
    #
    #     if configuration.loss == 'emd':
    #         epoch_loss /= len(train_data.point_clouds[0])
    #
    #     return epoch_loss, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})