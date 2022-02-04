from daisy import *
from cloudvolume import CloudVolume

TODO: DeprecationWarning

#TODO: Make callable function
# --- Parameters and paths --- #
in_file_path = 'gs://lee-pacureanu_data-exchange_us-storage/ls2892_LTP/2102/CBxs_lobV/CBxs_lobV_topm100um_30nm_rec_db12_.raw'
out_file_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/'
out_name = 'CBxs_lobV_topm100um_30nm_rec_db12_.n5'
ds_name = 'volumes/raw'

# --- Connect to image volume and get metadata --- #
print('Connecting to cloud volume...')

cloud_vol = CloudVolume(in_file_path, parallel=True)
voxel_size = Coordinate(cloud_vol.resolution)
offset = voxel_size*Coordinate(cloud_vol.voxel_offset)
shape = Coordinate(cloud_vol.volume_size)
total_roi = Roi(offset, voxel_size*shape)
dtype = cloud_vol.data_type

print('Opening a zarr volume...')
#modified daisy to not have to specify chunk_shape (line 319 in daisy/datasets.py)
vol = prepare_ds(out_file_path+out_name, 
                       ds_name,
                       total_roi,
                        voxel_size,
                        dtype)

# --- Load cloud and save local --- #
print('Downloading cloud volume...')
temp = cloud_vol[:,:,:].squeeze()

assert vol.shape == temp.shape, f"Zarr volume shape:{vol.shape} != Cloud volume shape: {temp.shape}"
print('Saving to zarr...')
vol[total_roi] = temp
print('Done.')