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


class LatentGAN(GAN):
    def __init__(self, name, learning_rate, n_output, noise_dim, discriminator, generator, beta=0.9, gen_kwargs={}, disc_kwargs={}, graph=None):

        self.noise_dim = noise_dim
        self.n_output = n_output
        self.discriminator = discriminator
        self.generator = generator
        self.learning_rate = learning_rate

        GAN.__init__(self, name, graph)

        with tf.variable_scope(name):

            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])                  # Noise vector.
            self.gt_data = tf.placeholder(tf.float32, shape=[None] + self.n_output)           # Ground-truth.
            self.z_data = tf.placeholder(tf.float32, shape=[None, noise_dim])
            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output)
            with tf.variable_scope('generator', reuse=True):
                self.generator_out_zdata = self.generator(self.z_data, self.n_output)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.gt_data, scope=scope)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope)
            self.loss_d = tf.reduce_mean(self.real_logit) - tf.reduce_mean(self.synthetic_logit)
            self.loss_g = tf.reduce_mean(self.synthetic_logit)
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = self.gt_data*epsilon + (1-epsilon)*self.generator_out
            with tf.variable_scope('discriminator') as scope:
                self.d_hat_prob, self.d_hat = self.discriminator(x_hat, reuse=True, scope=scope)
            gradients = tf.gradients(self.d_hat, x_hat)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
            self.loss_d += gradient_penalty
            #Post ICLR TRY: safe_log
            ##adding ae to latentgan code to test reconsgtructions
            # if(ae is not None):
            #     c = ae.configuration
            #     layer = c.decoder(self.generator_out, **c.decoder_args)
            #     if c.exists_and_is_not_none('close_with_tanh'):
            #         layer = tf.nn.tanh(layer)
            #
            #     self.gen_reconstr = tf.reshape(layer, [-1, ae.n_output[0], ae.n_output[1]])
            # ##

            self.loss_zdata = tf.reduce_mean(tf.reduce_sum(tf.square(self.generator_out_zdata-self.gt_data),axis=1))
            train_vars = tf.trainable_variables()
            self.z_grad = tf.gradients(self.loss_zdata, [self.z_data])
            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

            self.opt_d = self.optimizer(learning_rate, beta, self.loss_d, d_params)

            self.opt_g = self.optimizer(learning_rate, beta, self.loss_g, g_params)
            z_factor = 100.0
            self.opt_gz = self.optimizer(learning_rate, beta, self.loss_g+(self.loss_zdata/z_factor), g_params)
            #self.opt_gz = self.optimizer(learning_rate, beta, self.loss_g+(self.loss_zdata/z_factor), g_params)
            
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        z =np.random.normal(mu, sigma, (n_samples, ndims))
        #z = z *np.random.normal(0,0.1)
        # norm = np.sqrt(np.sum(z ** 2, axis=1))
        # norm = np.maximum(norm, np.ones_like(norm))

        # z = z / norm[:, np.newaxis]
        return z

    def _single_epoch_train(self, train_data, epoch, batch_size=50, noise_params={'mu':0, 'sigma':1},opt_gz=False, save_path = '../data/gan_model/latent_wgan64'):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        epoch_loss_z = 0.
        batch_size = batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in xrange(n_batches):
                feed, z_data, _, _ = train_data.next_batch(batch_size)

                # Update discriminator.
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)

                feed_dict = {self.gt_data: feed, self.noise: z, self.z_data:z_data}
                for i in range(4):
                    loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)

                if(opt_gz==False):
                    loss_g, _, z_grad,loss_z = self.sess.run([self.loss_g, self.opt_g, self.z_grad,self.loss_zdata], feed_dict=feed_dict)
                else:
                    loss_g, _, z_grad, loss_z = self.sess.run([self.loss_g, self.opt_gz, self.z_grad, self.loss_zdata],
                                                              feed_dict=feed_dict)
                
                    # z_update = z_data - self.learning_rate * z_grad[0]
                    # norm = np.sqrt(np.sum(z_update ** 2, axis=1))
                    # norm = np.maximum(norm,np.ones_like(norm))
                    # z_update_norm = z_update / norm[:, np.newaxis]
                    # z_data[:] = z_update_norm
                    # # print(str(np.sqrt(np.sum(z ** 2, axis=1))))
                    # print("z_data norm: "+  str(np.sqrt(np.sum(z_data ** 2, axis=1))))
                # Compute average loss
                epoch_loss_d += loss_d
                epoch_loss_g += loss_g
                epoch_loss_z +=loss_z

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        self.saver.save(self.sess, save_path, global_step = epoch)
        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        epoch_loss_z /=n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g,epoch_loss_z), duration

    def generate_lv(self, batch_size=50, noise_params={'mu':0, 'sigma':1}):
        z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)

        feed_dict = {self.noise: z}
        generator_out = self.sess.run([self.generator_out], feed_dict=feed_dict)
        return generator_out[0]

    def generate_lv_with_z(self, batch_size=50, noise_params={'mu': 0, 'sigma': 1},z_data=None):

        feed_dict = {self.noise: z_data}
        generator_out = self.sess.run([self.generator_out], feed_dict=feed_dict)
        return generator_out[0]
