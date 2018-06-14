'''
Created on April 27, 2017

@author: optas
'''
import numpy as np
import time
import tensorflow as tf
import pdb
from . gan import GAN

# from .. fundamentals.layers import safe_log
from tflearn import is_training
from latent_3d_points.src.IO import write_ply
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
            self.masked_cloud = tf.placeholder(tf.float32,shape=[None] + [None,3])

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output)

            with tf.variable_scope('discriminator') as scope:
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, scope=scope)

            # self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))
        self.loss_g = tf.reduce_mean(self.synthetic_logit)

        # zeros= t
        # self.loss_l2 = tf.reduce_mean(tf.square(self.generator_out-self.gt_data)*tf.cast(tf.greater(tf.abs(self.gt_data),0),tf.float32))
        self.loss_l2 = tf.reduce_mean(tf.square(self.generator_out-self.gt_data))

        # X_idx = tf.expand_dims(masked_cloud, axis=3)
        # X_diff = tf.reduce_sum(tf.square(X_idx - tf.expand_dims(tf.transpose(generator_out,[0,2,1]),axis=1)), axis=2)
        # X_diff_arg = tf.reduce_min(X_diff,axis=2)
        c = ae.configuration
        import pdb
        #pdb.set_trace()
        with tf.variable_scope(c.experiment_name,reuse=True):
            self.layer = c.decoder(self.generator_out, **c.decoder_args)
        if c.exists_and_is_not_none('close_with_tanh'):
            self.layer = tf.nn.tanh(self.layer)

        self.gen_reconstr = tf.reshape(self.layer, [-1, ae.n_output[0], ae.n_output[1]])

        #dist1, idx1, dist2, idx2 = nn_distance(self.masked_cloud, self.gen_reconstr)
        match = approx_match(self.masked_cloud, self.gen_reconstr)
        self.loss_chd = tf.reduce_mean(match_cost(self.masked_cloud, self.gen_reconstr, match))/2048
        #self.loss_chd = tf.reduce_mean(dist1)+tf.reduce_mean(dist2)
        self.loss_norm = 0.00001*0.0001*tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(self.noise),axis=1)-noise_dim))
        #Post ICLR TRY: safe_log

        train_vars = tf.trainable_variables()


        d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
        g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]
        self.noise_params = [v for v in train_vars if 'noise' in v.name]

        self.optim_cd, self.opt_cd = self.optimizer(learning_rate, beta, self.lc_wt *self.loss_g+self.loss_chd + 10*self.lc_wt*self.loss_l2 + self.loss_norm, self.noise_params)
        self.optim_g, self.opt_g = self.optimizer(learning_rate, beta, self.lc_wt *self.loss_g+self.loss_l2, self.noise_params)
        self.optim_l2, self.opt_l2 = self.optimizer(learning_rate, beta, 0*self.loss_g+self.lc_wt*self.loss_l2, self.noise_params) #ignoring loss g2
        self.saver = tf.train.Saver(d_params+g_params, max_to_keep=1)
        global_vars = tf.global_variables()
        self.adam_vars = [v for v in global_vars if 'Adam' in v.name or 'beta' in v.name]
        #pdb.set_trace()
        gan_train_params = d_params + g_params + self.noise_params + self.adam_vars
        # self.init = tf.global_variables_initializer()
        self.init = tf.variables_initializer(gan_train_params)

        # Launch the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if ae==None:
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)
        else:
            self.sess = ae.sess
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims)
)
    def _single_epoch_train(self, batch,masked_cloud, epoch, save_path = '../data/gan_model/',restore_epoch='99',lc_weight = 0.01):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        self.saver.restore(self.sess,save_path+'-'+restore_epoch)
        #return
        # n_examples = batch.num_examples
        epoch_loss_l2 = 0.
        epoch_loss_g = 0.
        start_time = time.time()
        # final_lc_weightt =0.01


        is_training(True, session=self.sess)
        # lc_wt_mat = [0.0001, 0.0005,0.001,0.005,0.01,0.05]
        lc_wt_mat = [0.001]#,0.1,0.5]
        g_losses= []
        l2_losses=[]
        chd_losses=[]
        norm_losses=[]

        # for i in xrange(epoch):
        #     feed_dict = {self.gt_data: batch, self.lc_wt: 5}
        #     loss_l2, _,noise_params = self.sess.run([self.loss_l2, self.opt_l2,self.noise_params[0]], feed_dict=feed_dict)
        #     print("l2_loss:" + str(loss_l2))

        for l in lc_wt_mat:
            # self.sess.run(tf.assign(self.noise_params[0],noise_params))
            self.sess.run(tf.variables_initializer(self.noise_params))

            try:
                # Loop over all batches
                is_training(True, session=self.sess)
                lc_wt = np.linspace(l, l/10.0, epoch)
                for i in xrange(20000):
                    #pdb.set_trace()
                    feed_dict = {self.gt_data: batch, self.lc_wt:lc_wt[i],self.masked_cloud:masked_cloud}
                    loss_g, loss_l2, _, loss_norm = self.sess.run([self.loss_g, self.loss_l2, self.opt_g, self.loss_norm], feed_dict=feed_dict)
                    if i%1000==0:
                        print("l2 loss:" + str(loss_l2) + " g_loss:" + str(loss_g)+ " loss norm:" + str(loss_norm))# + " loss chamfer:" + str(loss_chd) )
                reconstructions = self.sess.run(self.gen_reconstr, feed_dict=feed_dict)
                from sklearn.neighbors import NearestNeighbors as NN
                x_masked_recon=np.zeros(reconstructions.shape)
                pref = './recon_from_ac/'
                for k in range(5):
                    recons = reconstructions[k,:,:]
                    for pt in masked_cloud[k,:,:]:
                        nbrs = NN(n_neighbors=1,algorithm='kd_tree').fit(recons)
                        distances,indx = nbrs.kneighbors(np.expand_dims(pt,0))
                        recons = np.delete(recons,indx,0)
                    #pdb.set_trace()
                    x_masked_recon[k,:,:] = np.concatenate([masked_cloud[k,:,:],recons],axis=0)
                    write_ply(pref + "airplane_ae_" + str(0) + "_mixedmasked_" + str(k) + "_.ply", x_masked_recon[k, :, :])
                masked_cloud=x_masked_recon
                for i in xrange(epoch):
                    #pdb.set_trace()
                    feed_dict = {self.gt_data: batch, self.lc_wt:lc_wt[i],self.masked_cloud:masked_cloud}
                    loss_g, loss_l2, _,loss_chd, loss_norm = self.sess.run([self.loss_g, self.loss_l2, self.opt_cd,self.loss_chd, self.loss_norm], feed_dict=feed_dict)
                    if i%1000==0:
                        print("l2 loss:" + str(loss_l2) + " g_loss:" + str(loss_g) + " loss chamfer:" + str(loss_chd) + " loss norm:" + str(loss_norm))

                    # Compute average loss
                    epoch_loss_l2 += loss_l2
                    epoch_loss_g += loss_g

                cleaned_vector = self.sess.run(self.generator_out)
                is_training(False, session=self.sess)

            except Exception:
                raise
            g_losses.append(loss_g)
            l2_losses.append(loss_l2)
            chd_losses.append(loss_chd)
            norm_losses.append(loss_norm)


            save_path  ='cleaned_aefull_wgan_chd_' + str(l) +  '.txt'
            np.savetxt(save_path, cleaned_vector)
            print("cleaned vecs saved to "+save_path)

        print("final losses:")

        for i,l in enumerate(lc_wt_mat):
            print(l,"l2", l2_losses[i],"g_loss", g_losses[i],"chd_loss", chd_losses[i], "norm_losses",norm_losses[i])

        epoch_loss_d /= epoch
        epoch_loss_g /= epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
