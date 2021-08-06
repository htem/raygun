import daisy
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json
import scipy.ndimage as ndimage

config_f = sys.argv[1]
with open(config_f) as f:
    config = json.load(f)

hdf_ds = daisy.open_ds(
    config['hdf']['file'],
    config['hdf']['dataset'],
    )

dilation_steps = config['dilation_steps']
clear_mask_list = config['clear_mask_list']
# test_roi = daisy.Roi(
#     hdf_ds.roi.get_begin(),
#     # (hdf_ds.roi.get_shape()[0:3]),
#     (2, 2, 2, 1),
#     )
# print(test_roi)
# test_ndarray = hdf_ds[test_roi].to_ndarray()
# test_ndarray[0]

# voxel_size = hdf_ds.voxel_size[0:3]
voxel_size = daisy.Coordinate(config['hdf']['voxel_size'])
print(voxel_size)

roi = daisy.Roi(
    daisy.Coordinate((hdf_ds.roi.get_begin()[0:3]))*voxel_size,
    daisy.Coordinate((hdf_ds.roi.get_shape()[0:3]))*voxel_size,
    # (8, 8, 8),
    )
print(roi)

zarr_ds = daisy.prepare_ds(
    config['zarr']['file'],
    config['zarr']['dataset'],
    roi,
    voxel_size,
    np.uint8,
    # write_size=raw_dir_shape,
    # num_channels=self.out_dims,
    # temporary fix until
    # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
    # (we want gzip to be the default)
    compressor={'id': 'zlib', 'level': 5},
    delete=True
    )

data = hdf_ds[hdf_ds.roi].to_ndarray()[:, :, :, 0]

for i, s in enumerate(data):
    print(i)
    if i in clear_mask_list:
        print(f'Clear {i}')
        data[i] = 0
    else:
        data[i] = ndimage.binary_dilation(s, iterations=dilation_steps)

zarr_ds.data[:] = np.logical_not(data)
