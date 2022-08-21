import numpy as np
import daisy

TODO: DeprecationWarning

#TODO: Make callable function
# --- Parameters and paths --- #
volume_size = [3216, 3216, 3216]  # In voxels, xyz order
voxel_size = [90, 90, 90]  # In nm
total_roi = daisy.Roi((0,0,0), np.array(volume_size)*np.array(voxel_size))
in_file_path = '/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/'
out_file_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/'
# out_file_path = in_file_path
src_name = 'CBm_FN_lobX_90nm_tile2_rec_db9_twopass_quarterAngle.vol'
out_name = 'CBm_FN_lobX_90nm_tile2_rec_db9_twopass_full_.n5'
ds_name = 'volumes/quarterAngle'
dtype_in = np.float32
dtype_out = np.uint8

print(f'Opening a zarr volume')
#modified daisy to not have to specify chunk_shape (line 319 in daisy/datasets.py)
vol = daisy.prepare_ds(out_file_path+out_name, 
                       ds_name,
                       total_roi,
                        voxel_size,
                        dtype_out)

# --- Load image volume and upload --- #
print('Loading raw volume into memory...')
im = np.fromfile(in_file_path+src_name, dtype=dtype_in)
print('Reshaping...')
im = im.reshape(*volume_size, order='F').astype(dtype_out)

assert vol.shape == im.shape
print('Saving to zarr...')
vol[total_roi] = im
# vol[total_roi] = np.fromfile(in_file_path+src_name, dtype=dtype).reshape(*volume_size, order='F')
print('Done.')