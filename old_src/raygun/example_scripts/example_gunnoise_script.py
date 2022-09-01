import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from gunnoise import *

noise_version = '' # for making multiple independently generated noise versions (e.g. for Fourier Shell analysis)
src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/synapse/cb2/volumes/' # PATH FOR ZARR

raw_name = 'raw'
noise_dict = {'downX': 8, # cudegy mimic of 30nm pixel size (max uttained) from sensor at ESRF.i16a X-ray source, assuming 4nm voxel size EM source images
         'gaussBlur': 30, # cudegy mimic of 30nm resolution of KB mirrors at ESRF.i16a X-ray source
         'gaussNoise': None, # ASSUMES MEAN = 0, THIS SETS VARIANCE
         'poissNoise': True, # cudegy mimic of sensor shot noise (hot pixels) at ESRF.i16a X-ray source
        #  'deform': , # TODO: IMPLEMENT
         }

noise_order = [
                'gaussBlur', 
               'downX', 
               'gaussNoise', 
               'poissNoise'
               ]

samples = [
    # 'ml0', # should be already done
    # 'ml1',
    # 'cutout1',
    # 'cutout2',
    # 'cutout5',
    # 'cutout6',
    'cutout7',
    ]

print('Starting batch noising...')

noise_batch(samples,
    src_path,
    raw_name,
    noise_dict,
    noise_order)

print('Batch noising complete.')