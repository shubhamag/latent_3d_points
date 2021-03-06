
# coding: utf-8

# ## This notebook will help you train a vanilla Point-Cloud AE with the basic architecture we used in our paper.
#     (it assumes latent_3d_points is in the PYTHONPATH and the structural losses have been compiled)

# In[1]:


import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph

from latent_3d_points.src.IO import write_ply
import numpy as np
import pdb
# In[2]:
#
#
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
#

# Define Basic Parameters

# In[3]:


top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae'
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'
# class_name = raw_input('Give me the class name (e.g. "chair"): ').lower() #uncomment to ask class


# Load Point-Clouds

# In[4]:


# syn_id = snc_category_to_synth_id()[class_name]
# class_dir = osp.join(top_in_dir , syn_id)
##Test data
class_dir  = '/home/shubham/latent_3d_points/notebooks/unmasked'
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)


# Load default training parameters (some of which are listed beloq). For more details please print the configuration object.
#
#     'batch_size': 50
#
#     'denoising': False     (# by default AE is not denoising)
#
#     'learning_rate': 0.0005
#
#     'z_rotate': False      (# randomly rotate models of each batch)
#
#     'loss_display_step': 1 (# display loss at end of these many epochs)
#     'saver_step': 10       (# over how many epochs to save neural-network)

# In[5]:

#
# train_params = default_train_params()
#
#
# # In[7]:
#
#
# encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
# train_dir = create_dir(osp.join(top_out_dir, experiment_name))
#
#
# # In[9]:
#
#
# print enc_args
# print dec_args
#
#
# # In[10]:
# #
#
# conf = Conf(n_input = [n_pc_points, 3],
#             loss = ae_loss,
#             training_epochs = train_params['training_epochs'],
#             batch_size = train_params['batch_size'],
#             denoising = train_params['denoising'],
#             learning_rate = train_params['learning_rate'],
#             train_dir = train_dir,
#             loss_display_step = train_params['loss_display_step'],
#             saver_step = train_params['saver_step'],
#             z_rotate = train_params['z_rotate'],
#             encoder = encoder,
#             decoder = decoder,
#             encoder_args = enc_args,
#             decoder_args = dec_args
#            )
# conf.experiment_name = experiment_name
# conf.held_out_step = 5   # How often to evaluate/print out loss on
#                          # held_out data (if they are provided in ae.train() ).
# conf.save(osp.join(train_dir, 'configuration'))
#
#
# # Build AE Model.
#
# # In[11]:
#
#
# reset_tf_graph()
# ae = PointNetAutoEncoder(conf.experiment_name, conf)
#
#
# # Train the AE (save output to train_stats.txt)
#
# # In[1]:
# # ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/chair/',500)
# # ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/airplane/',800)
# # ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae/with_global_with_upsampling/trials',1)
# ae.restore_model('/home/shubham/latent_3d_points/data/single_class_ae',1000)
# print "Successfully loaded model"
#
#
#
# # Get a batch of reconstuctions and their latent-codes.
# reconstruct_from_latent_vectors = False
pref = "./masked/"
#
# if(reconstruct_from_latent_vectors == False):
feed_pc, feed_model_names, _ = all_pc_data.next_batch(5)
# in_copy = feed_pc.copy()
X = feed_pc
num_pts_removed = 1000
indx = np.random.randint(X.shape[1], size=X.shape[0])
temp = np.zeros(X.shape[:2])
temp[[np.arange(X.shape[0]), indx]]=1
X_idx = np.sum(X*np.expand_dims(temp, axis=2), axis=1, keepdims=True)
X_diff = np.sum(np.square(X_idx - X), axis=2)
X_diff_arg = np.argsort(X_diff,axis=1)
mask_inp = np.ones(X.shape[:2],dtype = np.float32)
mask_inp[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:, :num_pts_removed]]] = 0
mask_inp = np.expand_dims(mask_inp, axis=2)
X_masked = X[[np.expand_dims(np.arange(X.shape[0]), axis=1), X_diff_arg[:, num_pts_removed:]]]

# masked = mask_inp*X
masked = X_masked
in_copy = masked


#     # rmask = np.random.randint(2, size=in_copy.shape)
#     # in_copy = in_copy*rmask
write_ply(pref+"airplane0_masked.ply", in_copy[0])
write_ply(pref+"airplane1_masked.ply", in_copy[1])
write_ply(pref+"airplane2_masked.ply", in_copy[2])
write_ply(pref+"airplane3_masked.ply", in_copy[3])
write_ply(pref+"airplane4_masked.ply", in_copy[4])
#     # pdb.set_trace()
#
#
#     reconstructions = ae.reconstruct_with_mask(feed_pc)
#     # shape2 = reconstructions[0][2,:,:]
#     print "loss : " + str(reconstructions[1])
#
#     write_ply(pref+"airplane0_upsampled.ply",reconstructions[0][0,:,:])
#     write_ply(pref+"airplane1_upsampled.ply",reconstructions[0][1,:,:])
#     write_ply(pref+"airplane2_upsampled.ply",reconstructions[0][2,:,:])
#     write_ply(pref+"airplane3_upsampled.ply",reconstructions[0][3,:,:])
#     write_ply(pref+"airplane4_upsampled.ply",reconstructions[0][4,:,:])
#     # pdb.set_trace()
#     # print "reconstructed, shape:" + str(reconstructions.shape)
#     latent_codes = ae.transform(feed_pc)
# else:
#
#     lv_array  = np.loadtxt('/home/shubham/latent_3d_points/notebooks/cleaned_vector.txt')
#     lv_batch = lv_array
#
#     reconstructions = ae.decode(lv_batch)
#     for i in range(6):
#         write_ply("airplane" + str(i) + "_from_cleaned_lv.ply", reconstructions[i, :, :])
#
#

# Use any plotting mechanism such as matplotlib to visualize the results.
