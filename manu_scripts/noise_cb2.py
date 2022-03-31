# %%
import sys
sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
from boilerPlate import GaussBlur, Noiser

# SET CORRECT SCRIPT BELOW
from SplitCycleGun20220311XNH2EM_apply_cb2myelWM1_ import *
# %%
datapipe = cycleGun.datapipe_B

# noise_dict = {
#         # 'downX': 8, # cudegy mimic of 30nm pixel size (max uttained) from sensor at ESRF.i16a X-ray source, assuming 4nm voxel size EM source images
#         'gaussBlur': 30, # cudegy mimic of 30nm resolution of KB mirrors at ESRF.i16a X-ray source
#         'gaussNoise': None, # ASSUMES MEAN = 0, THIS SETS VARIANCE
#         'poissNoise': True, # cudegy mimic of sensor shot noise (hot pixels) at ESRF.i16a X-ray source
#          }

# noise_order = [
#                 'gaussBlur', 
#                'downX', 
#                'gaussNoise', 
#                'poissNoise'
#                ]

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
# %%
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

# %%
def test_noise(datapipe, 
            noise_order, 
            noise_dict, 
            test_size=(40, 2048, 2048)
            ):

    parts = [datapipe.source, 
            gp.RandomLocation(), 
            datapipe.reject, 
            datapipe.resample,
            datapipe.normalize_real
            ]
    pipeline = None
    for part in parts:
        if part is not None:
            pipeline = part if pipeline is None else pipeline + part
    
    pipeline, arrays, noise_name = bring_the_noise(datapipe.real, pipeline, noise_order, noise_dict)

    # request matching the model input and output sizes
    request = gp.BatchRequest()
    for array in arrays:
        request.add(array, test_size)
    
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    
    fig, axs = plt.subplots(1,len(arrays), figsize=(40, 40*len(arrays)))
    for i, array in enumerate(arrays):
        axs[i].imshow(batch[array].data.squeeze(), cmap='gray')
        axs[i].set_title(array.identifier)
    
    return batch, arrays, noise_name
# %%
