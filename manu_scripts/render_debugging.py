# %%
import runpy
import numpy as np
import daisy
import torch
from numpy import uint8
from scipy.signal.windows import tukey


# %%
side = 'B'
src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
src_name = 'volumes/raw_30nm'

# side = 'A'
# src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_0.n5'
# src_name = 'volumes/interpolated_90nm_aligned'

crop = 16

checkpoints = [310000, 320000, 330000, 340000, 350000]
script_base_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/'
scripts = [
    'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train.py',
    'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_train.py',
    'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2SplitNoBottle_train.py',
    'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2LinkNoBottle_train.py',
    'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-3SplitNoBottle_train.py',
    'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-3LinkNoBottle_train.py',    
]

script_path = script_base_path + scripts[0]
checkpoint = checkpoints[0]
side_length=None
cycle=False
smooth=True
num_workers=30
alpha=0.5

# %%
#Load Model
script_dict = runpy.run_path(script_path)

# %%
cycleGun = script_dict['cycleGun']
if side_length is None:
    side_length = cycleGun.side_length

#Set Dataset to Render
source = daisy.open_ds(src_path, src_name)
side = side.upper()
if type(checkpoints) is not list:
    checkpoints = [checkpoints]    

if len(cycleGun.model_name.split('-')) > 1:
    parts = cycleGun.model_name.split('-')
    label_prefix = parts[0] + parts[1][1:]
else:
    label_prefix = cycleGun.model_name
label_prefix += f'_seed{script_dict["seed"]}'

read_size = side_length
read_roi = daisy.Roi([0,0,0], source.voxel_size * read_size)
write_size = read_size

if crop:
    write_size -= crop*2

if smooth:
    window = torch.cuda.FloatTensor(tukey(write_size, alpha=alpha))
    window = (window[:, None, None] * window[None, :, None]) * window[None, None, :]
    chunk_size = int(write_size * (alpha / 2))
    write_pad = (source.voxel_size * write_size * (alpha / 2)) / 2
    write_size -= write_size * (alpha / 2)
    write_roi = daisy.Roi(source.voxel_size * crop + write_pad, source.voxel_size * write_size)
else:
    write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)
    chunk_size = int(write_size)

num_channels = 1
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }

net = 'netG1' if side == 'A' else 'netG2'
other_net = 'netG1' if side == 'B' else 'netG2'

# %%
this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{checkpoint}'
print(f"Loading checkpoint {this_checkpoint}...")
cycleGun.checkpoint = this_checkpoint
cycleGun.load_saved_model(this_checkpoint)
generator = getattr(cycleGun.model, net).cuda()
generator.eval()
print(f"Rendering checkpoint {this_checkpoint}...")

label_dict = {}
label_dict['fake'] = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}'
label_dict['cycled'] = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}{other_net}'

# %%
destination = daisy.prepare_ds(
            src_path, 
            label_dict['fake'],
            source.roi,
            source.voxel_size,
            source.dtype,
            write_size=write_roi.get_shape() if not smooth else write_pad*2,
            num_channels=num_channels,
            compressor=compressor,
            delete=True,
            # force_exact_write_size=True
            )

# %%
this_read = read_roi.shift(source.roi.get_begin())
this_write = this_read.grow(-source.voxel_size * crop, -source.voxel_size * crop)

# %%
data = source.to_ndarray(this_read)
data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
data -= np.iinfo(source.dtype).min
data /= np.iinfo(source.dtype).max
data *= 2
data -= 1
out = generator(data).detach().squeeze()
if crop:
    out = out[crop:-crop, crop:-crop, crop:-crop]
out += 1.0
out /= 2
if smooth:
    out *= window
out *= 255 #TODO: This is written assuming dtype = uint8
out = out.cpu().numpy().astype(uint8)
destination[this_write] = destination[this_write].to_ndarray() + out



# %%
