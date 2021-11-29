print('Importing dependencies...')

from functools import partial
import gunpowder as gp
from numpy.core.numeric import isclose
from boilerPlate import GaussBlur, Noiser
from scipy import optimize

# Load Trained Discriminator:
import sys
sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
from CycleGun_CBv30nmBottom100um_cb2gcl1_20211126_ import *

print('Defining functions...')
def bring_the_noise(src, pipeline, noise_order, noise_dict):
    this_array = src
    noise_name = ''
    arrays = [src]
    for noise in noise_order:
        noise_name += noise
        new_array = gp.ArrayKey(noise_name.upper())
        
        if noise == 'resample' and not isclose(noise_dict[noise]['ratio'], 1):
            pipeline += gp.Resample(this_array, noise_dict[noise]['base_voxel_size'] * noise_dict[noise]['ratio'], new_array)
        elif noise == 'gaussBlur' and not isclose(noise_dict[noise], 0):
            pipeline += GaussBlur(this_array, noise_dict[noise], new_array=new_array)
        elif noise == 'gaussNoise' and not isclose(noise_dict[noise], 0):
            pipeline += Noiser(this_array, new_array=new_array, mode='gaussian', var=noise_dict[noise])
        elif noise == 'poissNoise' and noise_dict[noise]:
            pipeline += Noiser(this_array, new_array=new_array, mode='poisson')
        elif 'noise' in noise and noise_dict[noise]:
            pipeline += Noiser(this_array, new_array=new_array, mode=noise_dict[noise]['mode'], **noise_dict[noise]['kwargs'])
        
        noise_name += '_'
        this_array = new_array
        arrays.append(new_array)
    
    noise_name = noise_name[:-1]
    return pipeline, arrays, noise_name

def update_noise_dict(noise_dict, optim_map, optim_vars):
    for var_keys, value in zip(optim_map, optim_vars):
        eval_str = "noise_dict"
        for key in var_keys:
            eval_str += f"['{key}']"
        eval_str += f" = {value}"
        exec(eval_str)
    return noise_dict

def cost_func(ref, pre_pipe, post_pipe, critic, out_array, noise_order, noise_dict, optim_map, optim_vars):
    # Setup Noising
    noise_dict = update_noise_dict(noise_dict, optim_map, optim_vars)
    pipe, _, _ = bring_the_noise(ref.real_B_src, pre_pipe, noise_order, noise_dict)
    pipe += post_pipe
    
    # Make Batch Request for Training
    request = gp.BatchRequest()
    extents = ref.get_extents()
    request.add(out_array, ref.common_voxel_size * extents, ref.common_voxel_size)

    # Get Batch
    with gp.build(pipe):
        batch = pipe.request_batch(request)

    # Noise(EM) should fake the discriminator
    pred_noised = critic(torch.as_tensor(batch[out_array].data))
    loss = ref.gan_loss(pred_noised, True)
    # print(f'Loss = {loss}')
    return loss.item()

print('Setting up pipeline parts...')

#####@@@@@ Setup Noise and other preferences
# noise_order = ['noise_speckle', 'noise_gauss', 'gaussBlur', 'resample', 'poissNoise']
noise_order = ['noise_speckle', 'gaussBlur', 'resample', 'poissNoise']
noise_dict = {
            'noise_speckle': 
                {
                    'mode': 'speckle',
                    'kwargs':
                        {
                            'mean': 0,
                            'var': 0.01
                        }
                },
            'gaussBlur': 6, # Sigma for guassian blur
            'resample': 
                {
                    'base_voxel_size': gp.Coordinate((4,4,4)),
                    'ratio': 3
                },
            'poissNoise': True
            }

optim_map = [
                [
                    'noise_speckle','kwargs','mean'
                ],
                [
                    'noise_speckle','kwargs','var'
                ],
                [
                    'gaussBlur'
                ],
                [
                    'resample','ratio'
                ]
            ]

optim_vars = [0, .01, 6, 3]

batch_size = 12

# cycleGun now stores a full CycleGAN model and pipeline
# cycleGun.netD2 is the Discriminator trained to differentiate real XNH from fake, we'll call it the "critic"
critic = cycleGun.netD2

# Get the source node for the EM
source = cycleGun.source_B

# Construct pipe
pre_pipe = source
pre_pipe += gp.RandomLocation()
pre_pipe += cycleGun.reject_B
pre_pipe += gp.Normalize(cycleGun.real_B_src)

noise_name = ''
for noise in noise_order:
    noise_name += noise
    noise_name += '_'
noise_name = noise_name[:-1]

# Add rest of pipe
out_array = gp.ArrayKey(noise_name.upper() + '_COMMONSIZE')
post_pipe = gp.Resample(gp.ArrayKey(noise_name.upper()), cycleGun.common_voxel_size, out_array)

# Add augmentations
# cycleGun.pipe_B += gp.SimpleAugment(mirror_only=augment_axes, transpose_only=augment_axes)


post_pipe += gp.Normalize(out_array)
#TODO: ADD CACHE

# cycleGun.pipe_B += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
#     control_point_spacing=cycleGun.side_length//2,
#     jitter_sigma=(10.0,)*cycleGun.ndims,
#     rotation_interval=(0, math.pi/8),
#     subsample=4,
#     spatial_dims=cycleGun.ndims
#     )


# add "channel" dimensions if neccessary, else use z dimension as channel
if cycleGun.ndims == len(cycleGun.common_voxel_size):
    post_pipe += gp.Unsqueeze([out_array])
# add "batch" dimensions
post_pipe += gp.Stack(batch_size)

func = partial(cost_func, cycleGun, pre_pipe, post_pipe, critic, out_array, noise_order, noise_dict, optim_map)
bounds = [
                [
                    None, None#'noise_speckle','kwargs','mean'
                ],
                [
                    0, None#'noise_speckle','kwargs','var'
                ],
                [
                    0, None#'gaussBlur' -sigma
                ],
                [
                    0.1, None#'resample','ratio'
                ]
            ]
options = {'disp': True} 

print('Optimizing...')
result = optimize.shgo(func, bounds, options=options)
print('Done.')
print(f'x = {result.x}, Final loss = {result.fun}, Total # local minima found = {len(result.xl)}')
