'''
Created on February 2, 2017

@author: optas
'''
import sys
import warnings
import os.path as osp
import tensorflow as tf
import numpy as np
import pdb
from tflearn import is_training

from . in_out import create_dir, pickle_data, unpickle_data
from . general_utils import apply_augmentations, iterate_in_chunks
from . neural_net import Neural_Net

model_saver_id = 'models.ckpt'


class Configuration():
    def __init__(self, n_input, encoder, decoder, encoder_args={}, decoder_args={},
                 training_epochs=200, batch_size=10, learning_rate=0.001, denoising=False,
                 saver_step=None, train_dir=None, z_rotate=False, loss='chamfer', gauss_augment=None,
                 saver_max_to_keep=None, loss_display_step=1, debug=False,
                 n_z=None, n_output=None, latent_vs_recon=1.0, consistent_io=None, adv_ae=False):

        # Parameters for any AE
        self.n_input = n_input
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug
        self.adv_ae = adv_ae

        # Used in VAE
        self.latent_vs_recon = np.array([latent_vs_recon], dtype=np.float32)[0]
        self.n_z = n_z

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

        self.consistent_io = consistent_io

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = self.__dict__.keys()
        vals = self.__dict__.values()
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').next()


class AutoEncoder(Neural_Net):
    '''Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.
    '''

    def __init__(self, name, graph, configuration):
        Neural_Net.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output
        self.train_counter = 0
        mask_inp = np.ones([configuration.batch_size, configuration.n_input[0], 1]) ##TODO uncomment for masking
        self.mask = tf.placeholder_with_default(mask_inp, [None, configuration.n_input[0], 1])
        configuration.encoder_args['mask']= self.mask

        in_shape = [None] + self.n_input
        out_shape = [None] + self.n_output

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, out_shape)
            else:
                self.gt = self.x

    def restore_model(self, model_path, epoch, verbose=True):
        '''Restore all the variables of a saved auto-encoder model.
        '''
        vars = tf.trainable_variables()
        vars_name = [var.name for var in vars]
        for i in range(len(vars_name)):
            print("\n"+vars_name[i])
        # sys.exit(0)
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print "Restored from :" + osp.join(model_path, model_saver_id + '-' + str(int(epoch)))
                print('Model restored in epoch {0}.'.format(epoch))

    def partial_fit_without_mask(self, X, GT=None,num_pts_removed = 1000):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        # print("training without mask")
        loss_d = 0

        try:
            if GT is not None:

                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT})
            else:
                if(self.train_counter==0):
                    print ("training WITHOUT mask")
                    self.train_counter+=1
                _, loss, recon,loss_g = self.sess.run((self.train_step, self.loss, self.x_reconstr,self.loss_g), feed_dict={self.x: X,})
                if (self.configuration.adv_ae == True):
                    loss = loss- loss_g
                    if not self.flag:
                        _, loss_d = self.sess.run((self.d_step, self.loss_d), feed_dict={self.x: X})
                    else:
                        loss_d = self.sess.run((self.loss_d), feed_dict={self.x: X})
                    #print(loss)
                    _, loss_d,g_grads,loss_g_grads = self.sess.run((self.d_step, self.loss_d,self.g_gradients,self.loss_g_gradients), feed_dict={self.x: X})
                    print("loss_g grads w.r.t g[0]",g_grads)
                    print("loss grads wrt g[0]",loss_g_grads)

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss,loss_d

    def partial_fit(self, X, GT=None,num_pts_removed = 1000,mask_type=0):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        is_training(True, session=self.sess)
        print("training with mask type " + str(mask_type))
        if(mask_type==0):
            print("error, partial_fit called with mask_type 0")
            exit()

        if(mask_type ==1):
            mask_inp = np.ones(X.shape[:2],dtype = np.float32)
            mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), np.random.choice(X.shape[1],[X.shape[0],num_pts_removed])]]=0
            mask_inp = np.expand_dims(mask_inp, axis=2)

        elif(mask_type==2):

            # mask = np.random.randint(2,size=X.shape)
            indx = np.random.randint(X.shape[1], size=X.shape[0])
            temp = np.zeros(X.shape[:2])
            temp[[np.arange(X.shape[0]), indx]]=1
            X_idx = np.sum(X*np.expand_dims(temp, axis=2), axis=1, keepdims=True)
            X_diff = np.sum(np.square(X_idx - X), axis=2)
            X_diff_arg = np.argsort(X_diff,axis=1)
            mask_inp = np.ones(X.shape[:2],dtype = np.float32)
            mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:, :num_pts_removed]]]=0
            mask_inp = np.expand_dims(mask_inp, axis=2)
        try:
            if GT is not None:

                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT, self.mask: mask_inp})
                if (self.configuration.adv_ae == True):
                    _, loss, recon = self.sess.run((self.d_step, self.loss_d, self.x_reconstr), feed_dict={self.x: X, self.gt: GT, self.mask: mask_inp})
            else:
                if(self.train_counter==0):
                    print ("training with random binary upsample mask")
                    self.train_counter+=1
                _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X, self.mask: mask_inp})
                if (self.configuration.adv_ae == True):
                    _, loss, recon = self.sess.run((self.d_step, self.loss_d, self.x_reconstr), feed_dict={self.x: X, self.mask: mask_inp})

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss



    # def discriminator(data, reuse=None, scope='disc'):
    #     with tf.variable_scope(scope, reuse=reuse):
    #         layer = tf.contrib.layers.fully_connected(data, 256)
    #         # layer = tf.contrib.layers.fully_connected(layer, 512)
    #         layer = tf.contrib.layers.fully_connected(layer, 128)
    #         layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
    #         prob = tf.nn.sigmoid(layer)
    #     return prob, layer
    def reconstruct(self, X, GT=None, compute_loss=True):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if compute_loss:
            loss = self.loss
        else:
            loss = tf.no_op()



        if GT is None:
            return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X })
        else:
            return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

    def reconstruct_with_mask(self, X, GT=None, compute_loss=True,num_pts_removed=1000,mask_type=0):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''

        print("reconstruct with mask called with mask type " + str(mask_type) +" num pts removed " + str(num_pts_removed))
        if compute_loss:
            loss = self.loss
        else:
            loss = tf.no_op()

        if (mask_type == 0):
            if GT is None:
                return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X})
            else:
                return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

        if mask_type == 1:

            mask_inp = np.ones(X.shape[:2], dtype=np.float32)
            sampled = np.random.choice(X.shape[1], [X.shape[0], num_pts_removed])
            mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), sampled]] = 0
            mask_inp = np.expand_dims(mask_inp, axis=2)
            # X_masked = X[[np.expand_dims(np.arange(X.shape[0]), axis=1), sampled]]
            if GT is None:
                return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.mask: mask_inp})
            else:
                return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})
        else:

            indx = np.random.randint(X.shape[1], size=X.shape[0])
            temp = np.zeros(X.shape[:2])
            temp[[np.arange(X.shape[0]), indx]] = 1
            X_idx = np.sum(X * np.expand_dims(temp, axis=2), axis=1, keepdims=True)
            X_diff = np.sum(np.square(X_idx - X), axis=2)
            X_diff_arg = np.argsort(X_diff, axis=1)
            mask_inp = np.ones(X.shape[:2], dtype=np.float32)
            mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:, :num_pts_removed]]] = 0
            mask_inp = np.expand_dims(mask_inp, axis=2)
            # X_masked = X[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:, num_pts_removed:]]]
            if GT is None:
                return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.mask: mask_inp})
            else:
                return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run([self.z,self.mask,self.noise], feed_dict={self.x: X})

    def transform_with_mask(self,X,num_pts_removed = 100,mask_type=0):
        print ("Transform with mask called, with mask type " +str(mask_type) + " "   + str(num_pts_removed) + " points removed")


        if(mask_type==0):
            return self.sess.run(self.z, feed_dict={self.x: X}), X

        if mask_type==1:


            mask_inp = np.ones(X.shape[:2],dtype = np.float32)
            sampled = np.random.choice(X.shape[1],[X.shape[0],num_pts_removed])
            mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), sampled]]=0
            mask_inp = np.expand_dims(mask_inp, axis=2)
            X_masked = X[[np.expand_dims(np.arange(X.shape[0]),axis=1),sampled]]

            return self.sess.run(self.z, feed_dict={self.x: X, self.mask: mask_inp}), X_masked
        else:

            indx = np.random.randint(X.shape[1], size=X.shape[0])
            temp = np.zeros(X.shape[:2])
            temp[[np.arange(X.shape[0]), indx]]=1
            X_idx = np.sum(X*np.expand_dims(temp, axis=2), axis=1, keepdims=True)
            X_diff = np.sum(np.square(X_idx - X), axis=2)
            X_diff_arg = np.argsort(X_diff,axis=1)
            mask_inp = np.ones(X.shape[:2],dtype = np.float32)
            mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:,:num_pts_removed]]]=0
            mask_inp = np.expand_dims(mask_inp, axis=2)
            X_masked = X[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:,num_pts_removed:]]]
            #pdb.set_trace()

            return self.sess.run(self.z, feed_dict={self.x: X,self.mask: mask_inp}),X_masked, X#, X_diff_arg

    def interpolate(self, x, y, steps):
        ''' Interpolate between and x and y input vectors in latent space.
        x, y np.arrays of size (n_points, dim_embedding).
        '''
        in_feed = np.vstack((x, y))
        z1, z2 = self.transform(in_feed.reshape([2] + self.n_input))
        all_z = np.zeros((steps + 2, len(z1)))

        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_z[i, :] = (alpha * z2) + ((1.0 - alpha) * z1)

        return self.sess.run((self.x_reconstr), {self.z: all_z})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def train(self, train_data, configuration, log_file=None, held_out_data=None,mask_type=0):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration,loss_d = self._single_epoch_train(train_data, c,mask_type=mask_type)
            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))
            stats.append((epoch, loss, duration,loss_d))

            if epoch % c.loss_display_step == 0:
                #pdb.set_trace()
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                print("loss_d   %4f"%(loss_d))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if held_out_data is not None and c.exists_and_is_not_none('held_out_step') and (epoch % c.held_out_step == 0):
                loss, duration,loss_d = self._single_epoch_train(held_out_data, c, only_fw=True,mask_type=mask_type)
                print("Held Out Data :", 'forward time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('On Held_Out: %04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))
        return stats

    def evaluate(self, in_data, configuration, ret_pre_augmentation=False):
        n_examples = in_data.num_examples
        data_loss = 0.
        pre_aug = None
        if self.is_denoising:
            original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
            if ret_pre_augmentation:
                pre_aug = feed_data.copy()
            if feed_data is None:
                feed_data = original_data
            feed_data = apply_augmentations(feed_data, configuration)  # This is a new copy of the batch.
        else:
            original_data, ids, _ = in_data.full_epoch_data(shuffle=False)
            feed_data = apply_augmentations(original_data, configuration)

        b = configuration.batch_size
        reconstructions = np.zeros([n_examples] + self.n_output)
        for i in xrange(0, n_examples, b):
            if self.is_denoising:
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b], original_data[i:i + b])
            else:
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b])

            # Compute average loss
            data_loss += (loss * len(reconstructions[i:i + b]))
        data_loss /= float(n_examples)

        if pre_aug is not None:
            return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data), pre_aug
        else:
            return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data)

    def evaluate_one_by_one(self, in_data, configuration):
        '''Evaluates every data point separately to recover the loss on it. Thus, the batch_size = 1 making it
        a slower than the 'evaluate' method.
        '''

        if self.is_denoising:
            original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
            if feed_data is None:
                feed_data = original_data
            feed_data = apply_augmentations(feed_data, configuration)  # This is a new copy of the batch.
        else:
            original_data, ids, _ = in_data.full_epoch_data(shuffle=False)
            feed_data = apply_augmentations(original_data, configuration)

        n_examples = in_data.num_examples
        assert(len(original_data) == n_examples)

        feed_data = np.expand_dims(feed_data, 1)
        original_data = np.expand_dims(original_data, 1)
        reconstructions = np.zeros([n_examples] + self.n_output)
        losses = np.zeros([n_examples])

        for i in xrange(n_examples):
            if self.is_denoising:
                reconstructions[i], losses[i] = self.reconstruct(feed_data[i], original_data[i])
            else:
                reconstructions[i], losses[i] = self.reconstruct(feed_data[i])

        return reconstructions, losses, np.squeeze(feed_data), ids, np.squeeze(original_data)

    def embedding_at_tensor(self, dataset, conf, feed_original=True, apply_augmentation=False, tensor_name='bottleneck'):
        '''
        Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
        Observation: the next layer after latent (z) might be something interesting.
        tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
        '''
        batch_size = conf.batch_size
        original, ids, noise = dataset.full_epoch_data(shuffle=False)

        if feed_original:
            feed = original
        else:
            feed = noise
            if feed is None:
                feed = original

        feed_data = feed
        if apply_augmentation:
            feed_data = apply_augmentations(feed, conf)

        embedding = []
        if tensor_name == 'bottleneck':
            for b in iterate_in_chunks(feed_data, batch_size):
                embedding.append(self.transform(b.reshape([len(b)] + conf.n_input)))
        else:
            embedding_tensor = self.graph.get_tensor_by_name(tensor_name)
            for b in iterate_in_chunks(feed_data, batch_size):
                codes = self.sess.run(embedding_tensor, feed_dict={self.x: b.reshape([len(b)] + conf.n_input)})
                embedding.append(codes)

        embedding = np.vstack(embedding)
        return feed, embedding, ids
