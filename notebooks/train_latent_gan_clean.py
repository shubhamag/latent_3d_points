import tensorflow as tf
import numpy as np
from latent_3d_points.src.latent_gan_clean import LatentGAN
import pdb



def discriminator(data, reuse=None, scope='disc'):
	with tf.variable_scope(scope, reuse=reuse):
		layer = tf.contrib.layers.fully_connected(data, 256)
		layer = tf.contrib.layers.fully_connected(layer, 512)
		#layer = tf.contrib.layers.fully_connected(layer, 128)
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


def GAN_cleaner(latent_vec=None,masked_cloud = None, ae=None,num_epochs=20000):

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
	latentgan._single_epoch_train(latent_vec,masked_cloud,epoch = num_epochs, save_path='/home/swami/deeprl/latent_3d_points/data/gan_model/wgan_ae_train',restore_epoch='1599')
	# import pdb
	# pdb.set_trace()
	feed_dict = None
	decodes,noise = latentgan.sess.run([latentgan.gen_reconstr,latentgan.noise],feed_dict=feed_dict)

	from latent_3d_points.src.IO import write_ply
	pref = './recon_from_ac/'
	# pdb.set_trace()
	for i in range(5):
		write_ply(pref + "airplane_wgan_train" + str(i) + "_.ply", decodes[i, :, :])
	# print("l2_loss %4f gen %4f duration %f"%(d_loss, g_loss, time))


if __name__ == '__main__':
	GAN_cleaner()
	
