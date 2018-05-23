

import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph

from latent_3d_points.src.IO import write_ply
import numpy as np
import math
import pdb


n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'




train_params = default_train_params()

encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
print enc_args
print dec_args


top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae/airplane_full'
train_dir = create_dir(osp.join(top_out_dir, experiment_name))



batch_size =int(10)
conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            # batch_size = train_params['batch_size'],
            batch_size = batch_size,
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


# Train the AE (save output to train_stats.txt) 

# In[1]:
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/chair/',500)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/clean/',410)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/car_train/',660)
ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full',600)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane_full_adv',600)


class_dir = '/home/shubham/latent_3d_points/data/airplane_full'
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

# latent_vec_file = '/home/shubham/latent_3d_points/data/' + "car_train_ae_train.txt"
latent_vec_file = '/home/shubham/latent_3d_points/data/' + "airplane_full_adv.txt"
# feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)

full_pc,_,_ = all_pc_data.full_epoch_data()
num_input = all_pc_data.num_examples


num_iters = int(math.ceil(num_input/float(batch_size)))
array_row_size = int(num_iters*batch_size)
print "lv num rows:" + str(array_row_size)
lv_array = np.zeros([ array_row_size, bneck_size])
for i in range(num_iters):
    feed_pc, feed_model_names, _ = all_pc_data.next_batch(batch_size)
    # latent_codes = ae.transform(feed_pc) ##also might want to switch to encoder_with_convs_and_symmetry in ae_template, tho not necessary###
    latent_codes,mask,noise = ae.transform(feed_pc)
    lv_array[i*batch_size:(i+1)*batch_size,:] = latent_codes


# np.savetxt(latent_vec_file,lv_array)
print ("Latent codes:")
print (str(latent_codes))
print(mask)
pdb.set_trace()

# pdb.set_trace()

reconstructions = ae.reconstruct(feed_pc)
# shape2 = reconstructions[0][2,:,:]
# print "loss : " + str(reconstructions[1])
write_ply("airplane_ae.ply",reconstructions[0][1,:,:])
write_ply("airplane_ae2.ply",reconstructions[0][2,:,:])
write_ply("airplane_ae3.ply",reconstructions[0][3,:,:])
# write_ply("airplane4.ply",reconstructions[0][4,:,:])
# pdb.set_trace()
# print "reconstructed, shape:" + str(reconstructions.shape)


# Use any plotting mechanism such as matplotlib to visualize the results.
