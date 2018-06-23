import tensorflow as tf
import numpy as np
from latent_3d_points.src.latent_gan_clean_v2 import LatentGAN
import pdb

import warnings


NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar


def linear(input_, output_size, name="linear", spectral_normed=True, update_collection=None, stddev=None, bias_start=0.0):

    shape = input_.get_shape().as_list()

    if stddev is None:
        stddev = np.sqrt(1. / (shape[1]))
    weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                             tf.truncated_normal_initializer(stddev=stddev))
    bias = tf.get_variable("b", [output_size],
                             initializer=tf.constant_initializer(bias_start))
    if spectral_normed:
      mul = tf.matmul(input_, spectral_normed_weight(weight, update_collection=update_collection))
    else:
      mul = tf.matmul(input_, weight)

    return mul + bias

def discriminator(data, reuse=None, scope='disc', update_collection=None):
    with tf.variable_scope(scope, reuse=reuse):
        layer = tf.contrib.layers.fully_connected(data, 256)
        layer = tf.contrib.layers.fully_connected(layer, 512)
        # layer = tf.contrib.layers.fully_connected(layer, 128)
        layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
        prob = tf.nn.sigmoid(layer)
    return prob, layer

# def discriminator(data, reuse=None, scope='disc', update_collection=tf.GraphKeys.UPDATE_OPS):
#     with tf.variable_scope(scope, reuse=reuse):
#         layer = tf.nn.relu(linear(data, 256, update_collection=update_collection))
#         layer = tf.nn.relu(linear(layer, 512, update_collection=update_collection))
#         # layer = tf.contrib.layers.fully_connected(layer, 128)
#         layer = linear(layer, 1, update_collection=update_collection)
#         prob = tf.nn.sigmoid(layer)
#     return prob, layer


def generator(noise, n_output):
	layer = tf.contrib.layers.fully_connected(noise, 128)
	layer = tf.contrib.layers.fully_connected(layer, n_output[0])
	return layer

# class latent_dataset:
# 	def __init__(self, data):
# 		self.num_examples = data.shape[0]
# 		self._index_in_epoch=0
# 		self.point_clouds = data
# 		self.epochs_completed = 0
#
# 	def next_batch(self, batch_size, seed=None):
# 		'''Return the next batch_size examples from this data set.
#         '''
# 		start = self._index_in_epoch
# 		self._index_in_epoch += batch_size
# 		if self._index_in_epoch > self.num_examples:
# 			self.epochs_completed += 1  # Finished epoch.
# 			self.shuffle_data(seed)  # shuffle data after each epoch
# 			# Start next epoch
# 			start = 0
# 			self._index_in_epoch = batch_size
# 		end = self._index_in_epoch
#
# 		return self.point_clouds[start:end], None, None
#
#
# 	def shuffle_data(self, seed=None):
# 		if seed is not None:
# 			np.random.seed(seed)
# 		perm = np.arange(self.num_examples)
# # 		np.random.shuffle(perm)
# # 		self.point_clouds = self.point_clouds[perm]
# 		return self


def GAN_cleaner(latent_vec=None,masked_cloud = None, ae=None, gt = None, num_epochs=20000):

	# latent_vec = np.loadtxt('/home/shubham/latent_3d_points/data/single_class_ae/clean/lv_with_mask_5.txt')
	# latent_vec = np.loadtxt('/home/shubham/latent_3d_points/notebooks/test_lvs.txt')
	# if(latent_vec is None):
	# 	latent_vec = np.loadtxt('/home/shubham/latent_3d_points/notebooks/gt_noisy_airplane_full.txt')

	bneck_size = latent_vec.shape[1]
	latent_vec = latent_vec[:10]
	batch_size = latent_vec.shape[0]
	if(masked_cloud is None):
		print("Error, masked clouds not given")
		# exit()
	# latent_vec_class = latent_dataset(latent_vec)
	latentgan = LatentGAN(name='latentgan', learning_rate=0.0001, n_output=[bneck_size], noise_dim=128,
						  discriminator=discriminator, generator=generator, beta=0.9, batch_size=batch_size, masked_cloud_size = masked_cloud.shape[1], ae=ae)

	#(d_loss, g_loss), time = latentgan._single_epoch_train(latent_vec,masked_cloud,epoch = num_epochs,
	latentgan._single_epoch_train(latent_vec,masked_cloud, gt, epoch = num_epochs, save_path='/home/swami/deeprl/latent_3d_points/data/gan_model/wgan_ae_train',restore_epoch='1599')
	# import pdb
	# pdb.set_trace()
	feed_dict = None
	decodes,noise = latentgan.sess.run([latentgan.gen_reconstr,latentgan.noise],feed_dict=feed_dict)

	from latent_3d_points.src.IO import write_ply
	pref = './recon_from_ac/'
	# pdb.set_trace()
	for i in range(5):
		write_ply(pref + "airplane_wgan_test" + str(i) + "_.ply", decodes[i, :, :])
	# print("l2_loss %4f gen %4f duration %f"%(d_loss, g_loss, time))


if __name__ == '__main__':
	GAN_cleaner()
	
