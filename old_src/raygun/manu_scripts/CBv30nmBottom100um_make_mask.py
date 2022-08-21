from daisy import *
import numpy as np
from tqdm import tqdm

# --- Parameters and paths --- #
file_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/'
file_name = 'CBxs_lobV_bottomp100um_30nm_rec_db9_.n5'
in_name = 'volumes/raw'
mask_name = 'volumes/volume_mask'
# below all in voxel units (not world units) *xyz
base = (1603, 1603, 0)
radius = 1475
height = 2048


# --- Connect to image volume and get metadata --- #
print('Loading volume metadata...')
vol = open_ds(file_path + file_name, in_name)

print('Opening a zarr volume for the mask...')
#modified daisy to not have to specify chunk_shape (line 319 in daisy/datasets.py)
mask_vol = prepare_ds(file_path+file_name, 
                        mask_name,
                        vol.roi,
                        vol.voxel_size,
                        bool)

# --- Make mask --- #
print('Making mask volume...')
# defining mask coordinate volume
mask = np.empty(vol.shape, dtype=bool) # too avoid memory issues
# defining circle
circle = np.fromfunction(lambda i, j: (i - base[0])**2 + (j - base[1])**2 <= radius**2, vol.shape[:2]).astype(np.bool)

# filling numpy mask
mask[:,:,base[2]:] = np.tile(circle, [height,1,1]).T

# check shape
assert vol.shape == mask.shape, f"Zarr volume shape:{vol.shape} != Mask volume shape: {mask.shape}"

# --- Save mask --- #
print('Saving to zarr...')
mask_vol[vol.roi] = mask
print('Done.')
