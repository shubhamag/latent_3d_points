
import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph

from latent_3d_points.src.IO import write_ply
import numpy as np
import pdb

# In[3]:


top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae/airplane_full_adv_g'
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'



train_params = default_train_params()


encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))



print enc_args
print dec_args

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))



reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)


# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/chair/',500)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane/',800)
ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full/',600)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full_adv_g/',600)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/',900)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/with_global_with_upsampling/',890)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/with_global_with_upsampling/trials',1)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/clean',410)
print "Successfully loaded model"



# Get a batch of reconstuctions and their latent-codes.
reconstruct_from_latent_vectors = True
pref = "./recon_from_ac/"

if(reconstruct_from_latent_vectors == False):
    class_dir = '/home/shubham/latent_3d_points/notebooks/gt/'
    all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

    feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
    # in_copy = feed_pc.copy()
    # rmask = np.random.randint(2, size=in_copy.shape[:2])
    # in_copy = in_copy*np.expand_dims(rmask,axis=2)
    # write_ply(pref+"airplane0_downsampled.ply", in_copy[0])
    # write_ply(pref+"airplane1_downsampled.ply", in_copy[1])
    # write_ply(pref+"airplane2_downsampled.ply", in_copy[2])
    # exit(0)
    # write_ply(pref+"airplane3_downsampled.ply", in_copy[3])
    # write_ply(pref+"airplane4_downsampled.ply", in_copy[4])
    # pdb.set_trace()


    # reconstructions = ae.reconstruct(feed_pc)
    reconstructions = ae.reconstruct_with_mask(feed_pc)
    # shape2 = reconstructions[0][2,:,:]
    print "loss : " + str(reconstructions[1])

    # write_ply(pref+"airplane0_acrecon_upsampling.ply",reconstructions[0][0,:,:])
    # write_ply(pref+"airplane1_acrecon_upsampling.ply",reconstructions[0][1,:,:])
    # write_ply(pref+"airplane2_acrecon_upsampling.ply",reconstructions[0][2,:,:])
    # write_ply(pref+"airplane3_acrecon.ply",reconstructions[0][3,:,:])
    # write_ply(pref+"airplane4_acrecon.ply",reconstructions[0][4,:,:])
    # # pdb.set_trace()
    # print "reconstructed, shape:" + str(reconstructions.shape)
    # latent_codes = ae.transform(feed_pc)

else:
    print "reconstructing from lvs"
    pref = './recon_from_ac/'

    # lv_array  = np.loadtxt('/home/shubham/latent_3d_points/notebooks/cleaned_vector_0.01.txt')
    # lv_array  = np.loadtxt('/home/shubham/latent_3d_points/notebooks/cleaned_vector_test_0.01.txt')
    # lv_array  = np.loadtxt('/home/shubham/latent_3d_points/notebooks/test_lvs.txt') ##directly use input vecs
    # lv_array  = np.loadtxt('/home/shubham/latent_3d_points/data/single_class_ae/clean/lv_with_mask_5.txt') ##noisy vecs
    lv_array  = np.loadtxt('cleaned_puregan2_0.5.txt') ##noisy vecs
    lv_batch = lv_array

    reconstructions = ae.decode(lv_batch)
    for i in range(5):
        write_ply(pref + "airplane_full_newgansglo_cleaned_" + str(i) + "_.ply", reconstructions[i, :, :])



# Use any plotting mechanism such as matplotlib to visualize the results.
