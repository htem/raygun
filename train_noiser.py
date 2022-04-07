# %%
print('Importing dependencies...')

from functools import partial
import gunpowder as gp
from numpy.core.numeric import isclose
from scipy import optimize
import json
import torch
import sys

sys.path.insert(0, '/n/groups/htem/users/jlr54/raygun/')
from boilerPlate import GaussBlur, Noiser

print('Defining functions...')
def bring_the_noise(src, pipeline, noise_order, noise_dict):
    this_array = src
    noise_name = ''
    arrays = [src]
    for noise in noise_order:
        noise_name += noise
        new_array = gp.ArrayKey(noise_name.upper())
        
        if noise == 'resample':# and not isclose(noise_dict[noise]['ratio'], 1):
            pipeline += gp.Resample(this_array, noise_dict[noise]['base_voxel_size'] * noise_dict[noise]['ratio'], new_array)
        elif noise == 'gaussBlur':# and not isclose(noise_dict[noise], 0):
            pipeline += GaussBlur(this_array, noise_dict[noise], new_array=new_array)
        elif noise == 'gaussNoise':# and not isclose(noise_dict[noise], 0):
            pipeline += Noiser(this_array, new_array=new_array, mode='gaussian', var=noise_dict[noise])
        elif noise == 'poissNoise':# and noise_dict[noise]:
            pipeline += Noiser(this_array, new_array=new_array, mode='poisson')
        elif 'noise' in noise:# and noise_dict[noise]:
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

def make_noise_pipe(ref, pre_pipe, post_pipe, out_array, noise_order, noise_dict, optim_map, optim_vars):
    # Setup Noising
    noise_dict = update_noise_dict(noise_dict, optim_map, optim_vars)
    pipe, arrays, noise_name = bring_the_noise(ref.real_src, pre_pipe, noise_order, noise_dict)
    arrays.append(out_array)
    pipe += post_pipe
    
    # Make Batch Request for Training
    request = gp.BatchRequest()
    extents = ref.get_extents(side_length=128)
    # for array in arrays:
    #     request.add(array, ref.common_voxel_size * extents)
    request.add(out_array, ref.common_voxel_size * extents)

    return pipe, request

def cost_func(ref, pre_pipe, post_pipe, critic, out_array, noise_order, noise_dict, optim_map, optim_vars):
    # Setup Noising
    pipe, request = make_noise_pipe(ref, pre_pipe, post_pipe, out_array, noise_order, noise_dict, optim_map, optim_vars)

    # Get Batch
    with gp.build(pipe):
        batch = pipe.request_batch(request)

    # Noise(EM) should fake the discriminator
    pred_noised = critic(torch.as_tensor(batch[out_array].data))
    loss = ref.gan_loss(pred_noised, True)
    # print(f'Loss = {loss}')
    return loss.item()

if __name__ == '__main__':    
    # Load Trained Discriminator:
    sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
    # from CycleGun_CBv30nmBottom100um_cb2gcl1_20211126_ import *
    from SplitCycleGun20220311XNH2EM_apply_cb2myelWM1_ import *

    # %%
    print('Setting up pipeline parts...')

    #####@@@@@ Setup Noise and other preferences
    # noise_order = ['noise_speckle', 'noise_gauss', 'gaussBlur', 'resample', 'poissNoise']
    noise_order = [
                    'noise_speckle', 
                    'gaussBlur', 
                    # 'resample', 
                    'poissNoise'
                    ]
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
                # 'resample': 
                #     {
                #         'base_voxel_size': gp.Coordinate((4,4,4)),
                #         'ratio': 3
                #     },
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
                    # [
                    #     'resample','ratio'
                    # ]
                ]

    optim_vars = [
                    0, 
                    .01, 
                    6, 
                    # 3
                ]

    batch_size = 3

    noise_name = ''
    for noise in noise_order:
        noise_name += noise
        noise_name += '_'
    noise_name = noise_name[:-1]

    # cycleGun now stores a full CycleGAN model and pipeline
    # cycleGun.netD2 is the Discriminator trained to differentiate real XNH from fake, we'll call it the "critic"
    critic = cycleGun.netD2

    # Get the source node for the EM
    datapipe = cycleGun.datapipe_B
    datapipe.get_extents = cycleGun.get_extents
    datapipe.common_voxel_size = cycleGun.common_voxel_size
    datapipe.gan_loss = cycleGun.loss.gan_loss

    # Construct pipe
    pre_parts = [datapipe.source, 
            gp.RandomLocation(), 
            datapipe.reject, 
            datapipe.normalize_real,
            datapipe.scaleimg2tanh_real
            ]
    pre_pipe = None
    for part in pre_parts:
        if part is not None:
            pre_pipe = part if pre_pipe is None else pre_pipe + part

    # Add rest of pipe
    out_array = gp.ArrayKey(noise_name.upper() + '_COMMONSIZE')
    post_pipe = gp.Resample(gp.ArrayKey(noise_name.upper()), cycleGun.common_voxel_size, out_array)

    #TODO: ADD CACHE

    # add "channel" dimensions if neccessary, else use z dimension as channel
    if cycleGun.ndims == len(cycleGun.common_voxel_size):
        post_pipe += gp.Unsqueeze([out_array])
    # add "batch" dimensions
    post_pipe += gp.Stack(batch_size)

    func = partial(cost_func, datapipe, pre_pipe, post_pipe, critic, out_array, noise_order, noise_dict, optim_map)
    bounds = [
                    [
                        -1, 1#'noise_speckle','kwargs','mean'
                    ],
                    [
                        0, 1#'noise_speckle','kwargs','var'
                    ],
                    [
                        0, 20#'gaussBlur' -sigma
                    ],
                    # [
                    #     0.1, 5#'resample','ratio'
                    # ]
                ]
    options = {'disp': True} 

    # %%
    print('Testing...')
    print(f'Initial loss: {func(optim_vars)}')

    # %%
    print('Optimizing...')
    # result = optimize.shgo(func, bounds, options=options)
    # result = optimize.basinhopping(func, optim_vars, niter=1000, disp=True)
    result = optimize.differential_evolution(func, bounds, disp=True)#, workers=10)
    print(f'x = {result.x}, Final loss = {result.fun}')#, Total # local minima found = {len(result.xl)}')

    print('Saving...')
    final_dict = update_noise_dict(noise_dict, optim_map, result.x)
    f = open("EM2XNH_noiseDict.json", "w")
    json.dump(final_dict, f)
    f.close()

    print('Done.')

    # %%
