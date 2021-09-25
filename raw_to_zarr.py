import numpy as np
import daisy


# --- Parameters and paths --- #
volume_size = [3216, 3216, 2048]  # In voxels, xyz order
voxel_size = [30, 30, 30]  # In nm
total_roi = daisy.Roi((0,0,0), np.array(volume_size)*np.array(voxel_size))
file_path = '/n/groups/htem/ESRF_id16a/LTP/cb_lobX_sept2020/'
src_name = 'CBxs_lobV_bottomp100um_30nm_rec_db9_.raw'
out_name = 'CBxs_lobV_bottomp100um_30nm_rec_db9_.n5'
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
print('Loading volume into memory...')
im = np.fromfile(file_path+src_name, dtype=dtype)
print('Reshaping...')
im = im.reshape(*volume_size, order='F')

assert vol.shape == im.shape
print('Saving to zarr...')
vol[total_roi] = im
print('Done.')