import tensorflow as tf
import numpy as np
from latent_3d_points.src.latent_gan_clean import LatentGAN

num_epoch = 100

def discriminator(data, reuse=None, scope='disc'):
	with tf.variable_scope(scope, reuse=reuse):
		layer = tf.contrib.layers.fully_connected(data, 256)
		layer = tf.contrib.layers.fully_connected(layer, 512)
		layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
		prob = tf.nn.sigmoid(layer)
	return prob, layer

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


def trainGAN():
	# latent_vec = np.loadtxt('/home/shubham/latent_3d_points/data/single_class_ae/clean/lv_with_mask_5.txt')
	# latent_vec = np.loadtxt('/home/shubham/latent_3d_points/notebooks/test_lvs.txt')
	latent_vec = np.loadtxt('/home/shubham/latent_3d_points/notebooks/gt_noisy_vecs_masked.txt')
	bneck_size = latent_vec.shape[1]
	latent_vec = latent_vec[:10]
	batch_size = latent_vec.shape[0]
	# latent_vec_class = latent_dataset(latent_vec)
	latentgan = LatentGAN(name='latentgan', learning_rate=0.0001, n_output=[bneck_size], noise_dim=64,
						  discriminator=discriminator, generator=generator, beta=0.9, batch_size=batch_size)


	(d_loss, g_loss), time = latentgan._single_epoch_train(latent_vec,epoch = 20000)
	print("l2_loss %4f gen %4f duration %f"%(d_loss, g_loss, time))


if __name__ == '__main__':
	trainGAN()
	