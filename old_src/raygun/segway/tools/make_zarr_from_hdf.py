import daisy
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json

config_f = sys.argv[1]
with open(config_f) as f:
    config = json.load(f)

hdf_ds = daisy.open_ds(
    config['hdf']['file'],
    config['hdf']['gt'],
    )

zarr_ds = daisy.prepare_ds(
    config['zarr']['file'],
    config['zarr']['gt'],
    hdf_ds.roi,
    hdf_ds.voxel_size,
    np.uint64,
    # write_size=raw_dir_shape,
    # num_channels=self.out_dims,
    # temporary fix until
    # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
    # (we want gzip to be the default)
    compressor={'id': 'zlib', 'level': 5}
    )

zarr_ds.data[:] = hdf_ds[hdf_ds.roi].to_ndarray()
