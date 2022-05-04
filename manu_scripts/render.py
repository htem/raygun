#%%
# import matplotlib.pyplot as plt
import daisy
import torch
import sys
sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
from CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train import *
# from CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_train import *
checkpoint = 350000

if len(cycleGun.model_name.split('-')) > 1:
    parts = cycleGun.model_name.split('-')
    label_prefix = parts[0] + parts[1][1:]
else:
    label_prefix = cycleGun.model_name
label_prefix += f'_seed{seed}'
label = f'{label_prefix}_checkpoint{checkpoint}_fake'
out_name = 'volumes/'+label
this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{checkpoint}'

cycleGun.load_saved_model(this_checkpoint)
generator = cycleGun.model.netG1.cuda()
generator.eval()
# del cycleGun # to conserve memory

#%%
path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
raw_name = 'volumes/interpolated_90nm_aligned'
source = daisy.open_ds(path, raw_name)
target_roi = source.data_roi.grow(source.voxel_size * -312, source.voxel_size * -312)
# target_roi = source.data_roi.grow(source.voxel_size * -462, source.voxel_size * -462)
data = source.to_ndarray(target_roi)
data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
data -= data.min()
data /= data.max()
data *= 2
data -= 1

# %%
out = generator(data).detach().squeeze()
out += 1.0
out /= 2
out *= 255
out = out.cpu().numpy().astype(source.dtype)

#%%
# fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# axs[0].imshow(data.squeeze()[data.shape[2]//2,...].cpu(), cmap='gray')
# axs[0].set_title('90nm interpolated')
# axs[1].imshow(out[out.shape[0]//2,...], cmap='gray')
# axs[1].set_title('Fake 30nm')

#%%
#Write result
out_ds = daisy.Array(out, target_roi, source.voxel_size)

chunk_size = 64
num_channels = 1
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }
num_workers = 30
write_size = out_ds.voxel_size * chunk_size
chunk_roi = daisy.Roi([0,]*len(target_roi.get_offset()), write_size)

destination = daisy.prepare_ds(
    path, 
    out_name,
    target_roi,
    out_ds.voxel_size,
    out.dtype,
    write_size=write_size,
    write_roi=chunk_roi,
    num_channels=num_channels,
    compressor=compressor)

#Prepare saving function/variables
def save_chunk(block:daisy.Roi):
    try:
        destination.__setitem__(block.write_roi, out_ds.__getitem__(block.read_roi))
        return 0 # success
    except:
        return 1 # error
        
#Write data to new dataset
success = daisy.run_blockwise(
            target_roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit='shrink',
            num_workers=num_workers,
            max_retries=2)

if success:
    print(f'{target_roi} from {path}/{raw_name} rendered and written to {path}/{out_name}')
else:
    print('Failed to save cutout.')

