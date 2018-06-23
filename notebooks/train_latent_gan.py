import tensorflow as tf
import numpy as np
#from latent_3d_points.src.latent_gan import LatentGAN
from latent_3d_points.src.vanilla_gan import LatentGAN


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

# def discriminator(data, reuse=None, scope='disc', update_collection=None):
#     with tf.variable_scope(scope, reuse=reuse):
#         layer = tf.contrib.layers.fully_connected(data, 256)
#         layer = tf.contrib.layers.fully_connected(layer, 512)
#         # layer = tf.contrib.layers.fully_connected(layer, 128)
#         layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
#         prob = tf.nn.sigmoid(layer)
#     return prob, layer

def discriminator(data, reuse=None, scope='disc', update_collection=tf.GraphKeys.UPDATE_OPS):
    with tf.variable_scope(scope, reuse=reuse):
        layer = tf.nn.relu(linear(data, 256, update_collection=update_collection))
        layer = tf.nn.relu(linear(layer, 512, update_collection=update_collection))
        # layer = tf.contrib.layers.fully_connected(layer, 128)
        layer = linear(layer, 1, update_collection=update_collection)
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
def eval_GAN(save_path,epoch,ae=None):
    import os.path as osp
    from latent_3d_points.src.autoencoder import Configuration as Conf
    from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
    from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
    from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
        load_all_point_clouds_under_folder
    from latent_3d_points.src.tf_utils import reset_tf_graph

    from latent_3d_points.src.IO import write_ply
    import numpy as np
    import pdb


    bneck_size = 128
    noise_dim_size = 64
    batch_size = 10
    # latentgan = LatentGAN(name = 'latentgan', learning_rate = 0.0001, n_output = [bneck_size],
    #                       noise_dim = noise_dim_size, discriminator = discriminator, generator = generator, beta=0.9)
    # latentgan.saver.restore(latentgan.sess,save_path+'-' + epoch)
    from latent_3d_points.src.latent_gan_clean import LatentGAN as LCGAN
    latentgan = LCGAN(name='latentgan', learning_rate=0.0001, n_output=[bneck_size], noise_dim=64,
                          discriminator=discriminator, generator=generator, beta=0.9, batch_size=batch_size,
                          masked_cloud_size=None, ae=ae)

    latentgan._single_epoch_train(None, None, epoch=10,
                                                           save_path='/home/shubham/latent_3d_points/data/gan_model/wgan_ae_full',
                                                           restore_epoch='794')
    #
    batch_size = 10
    #
    #
    input_noise = np.random.normal([batch_size,noise_dim_size])
    feed_dict = {latentgan.noise: input_noise}
    #
    gen_lv = latentgan.sess.run([latentgan.generator_out], feed_dict=feed_dict)
    pdb.set_trace()




    # if(ae is None):
    #     n_pc_points = 2048  # Number of points per model.
    #     bneck_size = 128  # Bottleneck-AE size
    #     ae_loss = 'emd'  # Loss to optimize: 'emd' or 'chamfer'
    #
    #     top_out_dir = '../data/'  # Use to save Neural-Net check-points etc.
    #     top_in_dir = '../data/shape_net_core_uniform_samples_2048/'  # Top-dir of where point-clouds are stored.
    #
    #     experiment_name = 'single_class_ae/airplane_full'
    #     train_params = default_train_params()
    #     encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    #     train_dir = create_dir(osp.join(top_out_dir, experiment_name))
    #     conf = Conf(n_input=[n_pc_points, 3],
    #                 loss=ae_loss,
    #                 # training_epochs = train_params['training_epochs'],
    #                 training_epochs=600,
    #                 batch_size=train_params['batch_size'],
    #                 denoising=train_params['denoising'],
    #                 learning_rate=train_params['learning_rate'],
    #                 train_dir=train_dir,
    #                 loss_display_step=train_params['loss_display_step'],
    #                 saver_step=train_params['saver_step'],
    #                 saver_max_to_keep=20,
    #                 z_rotate=train_params['z_rotate'],
    #                 encoder=encoder,k
    #                 decoder=decoder,
    #                 encoder_args=enc_args,
    #                 decoder_args=dec_args,
    #                 adv_ae=False
    #                 )
    #     conf.experiment_name = experiment_name
    #     reset_tf_graph()
    #     ae = PointNetAutoEncoder(conf.experiment_name, conf)
    #     ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full/', 600)
    np.savetxt("wgan_vecs.txt", gen_lv)


    pdb.set_trace()


    # reconstructions = ae.decode(gen_lv)

    # num_epoch =800
    # opt_gz= False
    # for i in xrange(num_epoch):
    #     (d_loss, g_loss,z_loss), time = latentgan._single_epoch_train(latent_vec_class,epoch = i,opt_gz=opt_gz,save_path=save_path)
    #     print("epoch %4d disc %4f gen %4f z_loss  %4f  duration %f"%(i,d_loss, g_loss, z_loss,time))
    #     # if(z_loss < 6.2 and cov_lv>0.6):
    #     #     print("Switching to opt gz")
    #     #     opt_gz=True
    #
    #
    #     ##VALIDATION##
    #     gen_lv = latentgan.generate_lv(batch_size=500)
    #     # gen_lv = latentgan.generate_lv_with_z(batch_size=500,z_data=latent_vec_class.z_data[:500])
    #     distances, indices = nbrs.kneighbors(gen_lv)
    #     print("vector distances mean =" + str(np.mean(distances)))
    #     unique_matches = np.unique(indices)
    #     cov_lv = unique_matches.shape[0] / float(500)
    #     print("coverage = " + str(cov_lv))


def trainGAN(ae=None):

    metrics_on_pcs = False

    if(metrics_on_pcs):
        from latent_3d_points.src.evaluation_metrics import minimum_matching_distance,coverage
    from sklearn.neighbors import NearestNeighbors

    latent_vec = np.loadtxt('/home/swami/deeprl/latent_3d_points/data/single_class_ae/airplane_train_ae.txt')
    save_path = '/home/swami/deeprl/latent_3d_points/data/gan_model/wgan_ae_train'

    bneck_size = latent_vec.shape[1]
    noise_dim_size = 128
    z_data = np.random.normal(0, 1, (latent_vec.shape[0], noise_dim_size))
    z_data = z_data * np.random.normal(0, 0.1)
    norm = np.sqrt(np.sum(z_data ** 2, axis=1))
    norm = np.maximum(norm,np.ones_like(norm))
    z_data = z_data / norm[:, np.newaxis]

    latent_vec_class = latent_dataset(latent_vec, z_data)
    latent_vec_class.shuffle_data()

    validation_batch_size= 500
    cov_lv =0
    latent_validation,_,_,_ = latent_vec_class.next_batch(batch_size=validation_batch_size)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(latent_validation)

    latentgan = LatentGAN(name = 'latentgan', learning_rate = 0.0001, n_output = [bneck_size], noise_dim = noise_dim_size, discriminator = discriminator, generator = generator, beta=0.9)

    num_epoch =1600
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
    #eval_GAN(save_path='/home/shubham/latent_3d_points/data/gan_model/wgan_ae_full',epoch='794')
