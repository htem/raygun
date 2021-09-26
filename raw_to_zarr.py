import numpy as np
import daisy

#TODO: Make callable function
# --- Parameters and paths --- #
volume_size = [3216, 3216, 3216]  # In voxels, xyz order
voxel_size = [90, 90, 90]  # In nm
total_roi = daisy.Roi((0,0,0), np.array(volume_size)*np.array(voxel_size))
file_path = '/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/'
src_name = 'CBm_FN_lobX_90nm_tile3_twopass_rec_.raw'
out_name = 'CBm_FN_lobX_90nm_tile3_twopass_rec_.n5'
ds_name = 'volumes/raw'
dtype=np.uint8

print(f'Opening a zarr volume')
#modified daisy to not have to specify chunk_shape (line 319 in daisy/datasets.py)
vol = daisy.prepare_ds(file_path+out_name, 
                       ds_name,
                       total_roi,
                        voxel_size,
                        dtype)

# --- Load image volume and upload --- #
print('Loading raw volume into memory...')
im = np.fromfile(file_path+src_name, dtype=dtype)
print('Reshaping...')
im = im.reshape(*volume_size, order='F')

assert vol.shape == im.shape
print('Saving to zarr...')
vol[total_roi] = im
print('Done.')