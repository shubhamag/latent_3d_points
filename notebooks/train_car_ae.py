
import os.path as osp
from .. src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from .. src.autoencoder import Configuration as Conf
from .. src.point_net_ae import PointNetAutoEncoder

from .. src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from .. src.tf_utils import reset_tf_graph


n_pc_points = 2048                # Number of points per model.
bneck_size = 128     # Bottleneck-AE size
ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'

top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
experiment_name = 'single_class_ae/car_train'


print ("training car with no mask")
train_params = default_train_params()


encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))


print enc_args
print dec_args



conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            # training_epochs = train_params['training_epochs'],
            training_epochs = 600,
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            saver_max_to_keep = 20,
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                         # held_out data (if they are provided in ae.train() ). ##use when training ae only on train data
conf.save(osp.join(train_dir, 'configuration'))



reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

class_dir = '/home/shubham/latent_3d_points/data/car_train'
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

test_dir = '/home/shubham/latent_3d_points/data/car_test'
all_test_data = load_all_point_clouds_under_folder(test_dir, n_threads=8, file_ending='.ply', verbose=True)

buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf,held_out_data=all_test_data, log_file=fout,mask_type =0)
fout.close()


