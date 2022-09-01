# %%
import numpy as np
from train_noiser import *
import daisy
import json

# Load CycleGAN object:
import sys
sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
from CycleGun_CBv30nmBottom100um_cb2gcl1_20220320SplitResSelu_train import *
# from SplitCycleGun20220311XNH2EM_apply_cb2myelWM1_ import *

# %%
print('Setting up pipeline parts...')
#Setup Noise and other preferences
noise_order = [
                'gaussBlur', 
                'noise_speckle', 
                'poissNoise'
                ]

with open("/n/groups/htem/users/jlr54/raygun/manu_scripts/EM2XNH_noiseDict.json", "r") as f:
    noise_dict = json.load(f)

noise_name = ''
for noise in noise_order:
    noise_name += noise
    noise_name += '_'
noise_name = noise_name[:-1]

# Get the source node for the EM
datapipe = cycleGun.datapipe_B
datapipe.get_extents = cycleGun.get_extents
datapipe.common_voxel_size = cycleGun.common_voxel_size
datapipe.gan_loss = cycleGun.loss.gan_loss

# Construct pipe
pre_parts = [datapipe.source, 
        datapipe.resample,
        datapipe.normalize_real, 
        datapipe.scaleimg2tanh_real
        ]
pre_pipe = None
for part in pre_parts:
    if part is not None:
        pre_pipe = part if pre_pipe is None else pre_pipe + part

# Add rest of pipe
out_array = gp.ArrayKey(noise_name.upper())

post_pipe = gp.IntensityScaleShift(out_array, 0.5, 0.5) # tanh to float32 image
post_pipe += gp.IntensityScaleShift(out_array, 255, 0) # float32 to uint8
post_pipe += gp.AsType(out_array, np.uint8)

pipe, scan_request = make_noise_pipe(datapipe, pre_pipe, post_pipe, out_array, noise_order, noise_dict)

# Declare new array to write to
compressor = {'id': 'blosc', 
            'clevel': 3,
            'cname': 'blosclz',
            'blocksize': 64
            }        
source_ds = daisy.open_ds(datapipe.src_path, datapipe.real_name)
datapipe.total_roi = source_ds.data_roi
write_size = scan_request[out_array].roi.get_shape()
daisy.prepare_ds(
    datapipe.out_path,
    'volumes/' + noise_name,
    datapipe.total_roi,
    daisy.Coordinate(cycleGun.common_voxel_size),
    np.uint8,
    write_size=write_size,
    num_channels=1,
    compressor=compressor)

pipe += gp.ZarrWrite(
        dataset_names = {out_array: 'volumes/' + noise_name},
        output_filename = datapipe.src_path,
        compression_type = compressor,
        #dataset_dtypes = {noisy: np.uint8} # save as 0-255 values (should match raw)
)

pipe += gp.Scan(scan_request, num_workers=16, cache_size=64)

# %%
if __name__ == '__main__':    
    print('Rendering...')
    with gp.build(pipe):
        pipe.request_batch(gp.BatchRequest())
    print('Done.')