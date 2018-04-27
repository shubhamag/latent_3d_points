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
    def __init__(self, name, learning_rate, n_output, noise_dim, discriminator, generator,lc_weight= 0.01, beta=0.9, batch_size=1, gen_kwargs={}, disc_kwargs={}, graph=None):

        self.noise_dim = noise_dim
        self.n_output = n_output
        self.discriminator = discriminator
        self.generator = generator

        GAN.__init__(self, name, graph)

        with tf.variable_scope(name):

            self.noise = tf.get_variable("noise", shape=[batch_size, noise_dim], initializer = tf.random_normal_initializer())                  # Noise vector.
            self.gt_data = tf.placeholder(tf.float32, shape=[None] + self.n_output)                                                           # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output)

            with tf.variable_scope('discriminator') as scope:
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, scope=scope)

            self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

            self.loss_l2 = tf.reduce_mean(tf.square(self.generator_out-self.gt_data))

            #Post ICLR TRY: safe_log

            train_vars = tf.trainable_variables()

            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]
            noise_params = [v for v in train_vars if 'noise' in v.name]
            self.opt_g = self.optimizer(learning_rate, beta, lc_weight *self.loss_g+self.loss_l2, noise_params)
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
    def _single_epoch_train(self, batch, epoch, batch_size=50, noise_params={'mu':0, 'sigma':1}, save_path = '../data/gan_model/gan_model',lc_weight = 0.01):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        self.saver.restore(self.sess,save_path+'-99')
        # n_examples = batch.num_examples
        epoch_loss_l2 = 0.
        epoch_loss_g = 0.
        start_time = time.time()

        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in xrange(epoch):
                feed_dict = {self.gt_data: batch}
                loss_g, loss_l2, _ = self.sess.run([self.loss_g, self.loss_l2, self.opt_g], feed_dict=feed_dict)

                # Compute average loss
                epoch_loss_l2 += loss_l2
                epoch_loss_g += loss_g

            cleaned_vector = self.sess.run(self.generator_out)
            is_training(False, session=self.sess)

        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)

        np.savetxt('cleaned_vector_' + str(lc_weight) +  '.txt', cleaned_vector)
        epoch_loss_d /= epoch
        epoch_loss_g /= epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
