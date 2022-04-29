# It is helpful to save scripts, such as this one, alongside the data in order
# to keep a complete record of the training/rendering process performed.
# Can be ran uninterrupted on server with: "nohup python training_script.py &"
# and monitored with: "watch tail nohup.out"
print('Importing dependencies...')
from functools import partial
import sys
raygun = '/n/groups/htem/users/jlr54/raygun/'
from raygun.CycleGAN import *


working_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/'
src_A_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_bottomp100um_30nm_rec_db9_.n5'
src_B_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/cb2/myelWM1.n5'
model_name = 'CycleGun_CBv30nmBottom100um_cb2gcl1_20220311SplitResSelu' 


print('Loading model...')
cycleGun = CycleGAN(src_A=src_A_path, #EXPECTS ZARR VOLUME
                src_B=src_B_path,
                A_voxel_size=gp.Coordinate((30, 30, 30)), #voxel_size of A
                B_voxel_size=gp.Coordinate((40, 32, 32)), #voxel_size of B
                common_voxel_size=gp.Coordinate((40, 32, 32)), #voxel size to cast all data into
                ndims=2,
                A_name='volumes/raw_sliced',
                mask_A_name='volumes/volume_mask',
                B_name='volumes/raw_mipmap3',
                batch_size=13,
                num_workers=16,
                cache_size=96,
                min_coefvar = 1e-02,
                adam_betas=[0.5, 0.999],

                loss_style = 'split',
                loss_kwargs = { 'l1_lambda': 42,
                                'identity_lambda': 1,
                                # 'gan_mode': 'wgangp',
                    },
                sampling_bottleneck=True,

                g_init_learning_rate=0.00004,
                gnet_type='resnet',
                gnet_kwargs={
                    'input_nc': 1,
                    'output_nc': 1,
                    'norm_layer': partial(torch.nn.InstanceNorm2d, affine=True),#, track_running_stats=True),
                    # 'padding_type': 'valid',     
                    'activation': torch.nn.SELU,
                    # 'add_noise': True,
                    'ngf': 64,
                    'n_blocks': 9, # resnet specific
                },

                d_init_learning_rate=0.00004,
                dnet_depth=4, # number of layers in Discriminator networks
                # d_downsample_factor=2,
                d_num_fmaps=64,
                d_kernel_size=3, 

                spawn_subprocess=True,
                side_length=200, # requires odd number for valid resnet9 (which gives odd output)
                model_name = model_name,
                model_path = working_path+'models/',
                tensorboard_path = working_path+'tensorboard/'+model_name,
                num_epochs = 100000,
                log_every=20,
                )              

# cycleGun.set_device(1)
print('Set up.')

if __name__ == '__main__':
    print(f"Rendering {src_B_path.split('/')[-1]} to pseudo-{src_A_path.split('/')[-1]}...")
    cycleGun.render_full(side_length=624, side='B', cycle=True)#, crop_to_valid=True)
    print(f"Rendered! And saved at {src_B_path}")
else:
    cycleGun.set_device(1)
    cycleGun.batch_size = 1
    cycleGun.build_machine()
    cycleGun.load_saved_model()
    print('Loaded.')
