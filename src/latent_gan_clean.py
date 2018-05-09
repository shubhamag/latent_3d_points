'''
Created on April 27, 2017

@author: optas
'''
import numpy as np
import time
import tensorflow as tf

from . gan import GAN

# from .. fundamentals.layers import safe_log
from tflearn import is_training
try:    
    from .. external.structural_losses.tf_nndistance import nn_distance
    from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')



class LatentGAN(GAN):
    def __init__(self, name, learning_rate, n_output, noise_dim, discriminator, generator,lc_weight= 0.001, beta=0.9, batch_size=1, gen_kwargs={}, disc_kwargs={}, graph=None,masked_cloud_size=1024, ae=None):

        self.noise_dim = noise_dim
        self.n_output = n_output
        self.discriminator = discriminator
        self.generator = generator
        self.masked_cloud_size= masked_cloud_size
        # self.final_lc_weight = 0.01

        GAN.__init__(self, name, graph)

        with tf.variable_scope(name):

            self.noise = tf.get_variable("noise", shape=[batch_size, noise_dim], initializer = tf.random_normal_initializer())                  # Noise vector.
            self.gt_data = tf.placeholder(tf.float32, shape=[None] + self.n_output)                                                           # Ground-truth.
            self.lc_wt = tf.placeholder(tf.float32)  # Ground-truth.
            self.masked_cloud = tf.placeholder(tf.float32,shape=[None] + [self.masked_cloud_size,3])

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output)

            with tf.variable_scope('discriminator') as scope:
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, scope=scope)

            # self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))
            self.loss_g = tf.reduce_mean(self.synthetic_logit)

            # zeros= t
            # self.loss_l2 = tf.reduce_mean(tf.square(self.generator_out-self.gt_data)*tf.cast(tf.greater(tf.abs(self.gt_data),0),tf.float32))
            #self.loss_l2 = tf.reduce_mean(tf.square(self.generator_out-self.gt_data))

            # X_idx = tf.expand_dims(masked_cloud, axis=3)
            # X_diff = tf.reduce_sum(tf.square(X_idx - tf.expand_dims(tf.transpose(generator_out,[0,2,1]),axis=1)), axis=2)
            # X_diff_arg = tf.reduce_min(X_diff,axis=2)
            c = ae.configuration
            layer = c.decoder(self.generator_out, **c.decoder_args)
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.gen_reconstr = tf.reshape(layer, [-1, ae.n_output[0], ae.n_output[1]])
            
            dist, idx, _, _ = nn_distance(self.masked_cloud, self.gen_reconstr)
            self.loss_l2 = tf.reduce_mean(dist)
            #Post ICLR TRY: safe_log

            train_vars = tf.trainable_variables()

            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]
            noise_params = [v for v in train_vars if 'noise' in v.name]
            self.opt_g = self.optimizer(learning_rate, beta, self.lc_wt *self.loss_g+self.loss_l2, noise_params)
            self.saver = tf.train.Saver(d_params+g_params, max_to_keep=1)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)
            

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims)
)
    def _single_epoch_train(self, batch,masked_cloud, epoch, save_path = '../data/gan_model/latent_wgan64',lc_weight = 0.01):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        self.saver.restore(self.sess,save_path+'-99')
        # n_examples = batch.num_examples
        epoch_loss_l2 = 0.
        epoch_loss_g = 0.
        start_time = time.time()
        # final_lc_weightt =0.01


        is_training(True, session=self.sess)
        lc_wt_mat = [0.0001, 0.0005,0.001,0.005,0.01,0.05]
        g_losses= []
        l2_losses=[]
        for l in lc_wt_mat:
            self.sess.run(tf.global_variables_initializer())

            try:
                # Loop over all batches
                is_training(True, session=self.sess)
                lc_wt = np.linspace(l, l/10.0, epoch)
                for i in xrange(epoch):

                    feed_dict = {self.gt_data: batch, self.lc_wt:lc_wt[i],self.masked_cloud:masked_cloud}
                    loss_g, loss_l2, _ = self.sess.run([self.loss_g, self.loss_l2, self.opt_g], feed_dict=feed_dict)
                    print("l2 loss:" + str(loss_l2) + " g_loss:" + str(loss_g))

                    # Compute average loss
                    epoch_loss_l2 += loss_l2
                    epoch_loss_g += loss_g

                cleaned_vector = self.sess.run(self.generator_out)
                is_training(False, session=self.sess)

            except Exception:
                raise
            g_losses.append(loss_g)
            l2_losses.append(loss_l2)


            save_path  ='gt_wgan64_cleaned_' + str(l) +  '.txt'
            np.savetxt(save_path, cleaned_vector)
            print("cleaned vecs saved to "+save_path)

        print("final losses:")

        for i,l in enumerate(lc_wt_mat):
            print(l,l2_losses[i],g_losses[i])

        epoch_loss_d /= epoch
        epoch_loss_g /= epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
