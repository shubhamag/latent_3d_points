import tensorflow as tf
import numpy as np
from latent_3d_points.src.latent_gan import LatentGAN
import pdb



def discriminator(data, reuse=None, scope='disc'):
    with tf.variable_scope(scope, reuse=reuse):
        layer = tf.contrib.layers.fully_connected(data, 256)
        # layer = tf.contrib.layers.fully_connected(layer, 512)
        layer = tf.contrib.layers.fully_connected(layer, 128)
        layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
        prob = tf.nn.sigmoid(layer)
    return prob, layer

def generator(noise, n_output):
    layer = tf.contrib.layers.fully_connected(noise, 128)
    layer = tf.contrib.layers.fully_connected(layer, n_output[0])
    return layer

class latent_dataset:

    def __init__(self, data, z_data):
        self.num_examples = data.shape[0]
        self._index_in_epoch=0
        self.point_clouds = data
        self.epochs_completed = 0
        self.z_data = z_data

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)  # shuffle data after each epoch
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        return self.point_clouds[start:end], self.z_data[start:end], None, None


    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.z_data = self.z_data[perm]
        return self


# def trainGAN(ae=None):
#
#
#
#     latent_vec = np.loadtxt('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full_ae_full.txt')
#
#
#
#     bneck_size = latent_vec.shape[1]
#     latent_vec_class = latent_dataset(latent_vec)
#     latent_vec_class.shuffle_data()
#
#     validation_batch_size= 100
#     latent_validation,_,_ = latent_vec_class.next_batch(batch_size=validation_batch_size)
#     from sklearn.neighbors import NearestNeighbors
#
#     print("training gan on " +str(latent_vec_class.num_examples) + " latent vecs")
#     latentgan = LatentGAN(name = 'latentgan', learning_rate = 0.0001, n_output = [bneck_size], noise_dim = 64,
#                           discriminator = discriminator, generator = generator, beta=0.9)
#     num_epoch = 110
#
#     for i in xrange(num_epoch):
#         (d_loss, g_loss), time = latentgan._single_epoch_train(latent_vec_class,epoch = i,save_path=save_path)
#         print("disc %4f gen %4f duration %f"%(d_loss, g_loss, time))



def trainGAN(ae=None):

    metrics_on_pcs = False

    if(metrics_on_pcs):
        from latent_3d_points.src.evaluation_metrics import minimum_matching_distance,coverage
    from sklearn.neighbors import NearestNeighbors

    latent_vec = np.loadtxt('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full.txt')
    save_path = '../data/gan_model/latent_wganlgo2_airplane_full'

    bneck_size = latent_vec.shape[1]
    noise_dim_size = 32
    z_data = np.random.normal(0, 1, (latent_vec.shape[0], noise_dim_size))
    z_data = z_data * np.random.normal(0, 0.1)
    norm = np.sqrt(np.sum(z_data ** 2, axis=1))
    norm = np.maximum(norm,np.ones_like(norm))
    z_data = z_data / norm[:, np.newaxis]

    latent_vec_class = latent_dataset(latent_vec, z_data)
    latent_vec_class.shuffle_data()

    validation_batch_size= 4000
    cov_lv =0
    latent_validation,_,_,_ = latent_vec_class.next_batch(batch_size=validation_batch_size)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(latent_validation)


    latentgan = LatentGAN(name = 'latentgan', learning_rate = 0.0001, n_output = [bneck_size], noise_dim = noise_dim_size, discriminator = discriminator, generator = generator, beta=0.9)

    num_epoch =800
    opt_gz= False
    for i in xrange(num_epoch):
        (d_loss, g_loss,z_loss), time = latentgan._single_epoch_train(latent_vec_class,epoch = i,opt_gz=opt_gz,save_path=save_path)
        print("epoch %4d disc %4f gen %4f z_loss  %4f  duration %f"%(i,d_loss, g_loss, z_loss,time))
        # if(z_loss < 6.2 and cov_lv>0.6):
        #     print("Switching to opt gz")
        #     opt_gz=True


        ##VALIDATION##
        gen_lv = latentgan.generate_lv(batch_size=500)
        # gen_lv = latentgan.generate_lv_with_z(batch_size=500,z_data=latent_vec_class.z_data[:500])
        distances, indices = nbrs.kneighbors(gen_lv)
        print("vector distances mean =" + str(np.mean(distances)))
        unique_matches = np.unique(indices)
        cov_lv = unique_matches.shape[0] / float(500)
        print("coverage = " + str(cov_lv))

        if (metrics_on_pcs):
            reconstructions = ae.decode(gen_lv)
            cov = coverage(reconstructions, validation_pcs)

if __name__ == '__main__':
    trainGAN()