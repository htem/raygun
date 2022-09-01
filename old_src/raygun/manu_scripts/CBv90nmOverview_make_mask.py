from distutils.dir_util import remove_tree
import os
import tempfile
from daisy import *
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils/')
from wkw_seg_to_zarr import *

# --- Parameters and paths --- #
file_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/'
file_name = 'CBxs_lobV_overview_90nm_rec5iter_db9_l20p15_.n5'
in_name = 'volumes/raw'
mask_name = 'volumes/training_mask'
# # below all in voxel units (not world units) *xyz
# base = (1604, 1604, 0)
# radius = 1594
# height = 3216

mask_rois = [
    Roi((540, 450, 720), (1080, 1080,  720)),
    Roi((810, 810,   0), (450, 630, 450))
]

annotation_ID = '62ded548010000b200fe6375'


# --- Connect to image volume and get metadata --- #
file = os.path.join(file_path, file_name)
mask = os.path.join(file, mask_name)
if os.path.exists(mask):
    print('Deleting old mask...')
    remove_tree(mask)
# print('Loading volume metadata...')
# vol = open_ds(file, in_name)

# print('Opening a zarr volume for the mask...')
# #modified daisy to not have to specify chunk_shape (line 319 in daisy/datasets.py)
# mask_vol = prepare_ds(file, 
#                         mask_name,
#                         vol.roi,
#                         vol.voxel_size,
#                         bool)

# --- Make mask --- #
print('Making mask volume...')
# defining mask coordinate volume
# mask = np.empty(vol.shape, dtype=bool) # too avoid memory issues

# # defining circle
# circle = np.fromfunction(lambda i, j: (i - base[0])**2 + (j - base[1])**2 <= radius**2, vol.shape[:2]).astype(np.bool)

# # filling numpy mask
# mask[:,:,base[2]:] = np.tile(circle, [height,1,1]).T

# check shape
# assert vol.shape == mask.shape, f"Zarr volume shape:{vol.shape} != Mask volume shape: {mask.shape}"

# --- Get Webknossos mask --- #
with tempfile.TemporaryDirectory() as save_path:
    mask_vol = get_wk_mask(annotation_ID, save_path, file, in_name, save_name=mask_name)

for mask_roi in mask_rois:
    mask_vol[mask_roi] = False

print('Done.')
