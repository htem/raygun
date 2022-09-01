# %% [markdown]
# ## Imports
import numpy as np
from matplotlib import pyplot as plt
import zarr
import os

import gunpowder as gp
# import logging
# logging.basicConfig(level=logging.INFO)

# from this repo
from boilerPlate import GaussBlur, Noiser
from gunnoise import *

# %% [markdown]
# # Specify Parameters (source, noise type, downsampling, etc.)

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

# noise_order = ['downX', 
#                'gaussBlur', 
#                'gaussNoise', 
#                'poissNoise'
#                ]

samples = [
    'ml0', # should be already done
    'ml1',
    'cutout1',
    'cutout2',
    'cutout5',
    'cutout6',
    'cutout7',
    ]

src_voxel_size = (40, 4, 4)

# %% [markdown]
# # Check out raw data

# %%
sample = samples[0]
src = f'{src_path}{sample}/{sample}.zarr/volumes'
data = zarr.open(src)

# %%
#pick indices/window
window = 256
x_off = 100
y_off = 100
z_off = 10
plt.imshow(data['raw'][z_off, y_off:y_off+window, x_off:x_off+window], cmap='gray')

# %% [markdown]
# # Setup Noising Pipeline

# %% [markdown]
# ### Test pipeline before saving:

# %%
noise_batch([samples[0]],
    src_path,
    raw_name,
    noise_dict,
    noise_order,
    src_voxel_size=(40, 4, 4),
    check_every=250,
    scan_size=(40, 512, 512)
    )

# %%
sample = samples[0]
test_batch, arrays, noise_name = test_noise(sample,
                    src_path,
                    raw_name,
                    noise_dict,
                    noise_order,
                    src_voxel_size=(40, 4, 4),
                    test_size=(40, 2048, 2048)
                    )

# %% [markdown]
# ### Run actual

# %%
#in case you want to check in on the batch outputs:
raw = arrays[0]
noisy = arrays[-1]
test_batch[noisy]

# %%
# Check outputs
sample = samples[0]
src = f'{src_path}{sample}/{sample}.zarr/volumes'
data = zarr.open(src)

# %%
#pick indices/window
window = 256
x_off = 100
y_off = 100
z_off = 10
plt.imshow(data[noise_name][z_off, y_off:y_off+window, x_off:x_off+window], cmap='gray')

# %%
test = data[noise_name]
test
# data['raw']
# sample

# %%


# %%



