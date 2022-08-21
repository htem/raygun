import daisy
from daisy import Coordinate, Roi
import numpy as np
import sys
import json
import os
# from funlib.segment.arrays import replace_values

import gt_tools

'''
0 = myelin
255 = no myelin
'''

config = gt_tools.load_config(sys.argv[1])
# with open(config_f) as f:
    # config = json.load(f)

file = config["file"]

mask_ds = daisy.open_ds(
    file,
    config["mask_ds"])

raw_file = config["raw_file"]
out_file = config["out_file"]
# script_name = config["script_name"]
# script_name = script_name.split(".")[0]
# raw_file = config["zarr"]["dir"] + "/" + script_name + ".zarr"
# if "raw_file" in config:
#     raw_file = config["raw_file"]

# raw_ds = daisy.open_ds(
#     raw_file,
#     "volumes/raw")

# total_roi = raw_ds.roi

# script_name = os.path.basename(config_f)
# script_name = script_name.split(".")[0]
# out_file = config["zarr"]["dir"] + "/" + script_name + ".zarr"

out = daisy.prepare_ds(
    out_file,
    "volumes/labels/myelin_gt",
    mask_ds.roi,
    mask_ds.voxel_size,
    mask_ds.dtype,
    compressor={'id': 'zlib', 'level': 5}
    )

out_array = daisy.Array(
    np.zeros(out.shape, dtype=out.dtype), out.roi, out.voxel_size)

if "myelin_mask_file" in config:

    print("Reading myelin...")
    myelin_ds = daisy.open_ds(config["myelin_mask_file"], "exported_data")
    downsample_xy = config["myelin_mask_downsample_xy"]
    myelin_ndarray = myelin_ds.to_ndarray()
    myelin_ndarray = myelin_ndarray[:, :, :, 0]
    np.place(myelin_ndarray, myelin_ndarray == 1, 0)
    np.place(myelin_ndarray, myelin_ndarray == 2, 255)

    print("Upsampling myelin...")
    y_axis = 1
    x_axis = 2
    myelin_ndarray = np.repeat(myelin_ndarray, downsample_xy, axis=y_axis)
    myelin_ndarray = np.repeat(myelin_ndarray, downsample_xy, axis=x_axis)

    out_array[mask_ds.roi] = myelin_ndarray

elif os.path.exists(os.path.join(file, config["myelin_ds"])):

    # opening predicted myelin
    myelin_ds = daisy.open_ds(file, config["myelin_ds"])
    out_array[mask_ds.roi] = myelin_ds

    # sometimes myelin prediction is not too good, so we use label == 0
    # as a heuristic for getting myelin
    segment_ds = daisy.open_ds(file, config["segment_ds"])
    out_ndarray = out_array.to_ndarray()
    np.place(out_array.to_ndarray(), segment_ds.to_ndarray() == 0, 0)
    out_array[out_array.roi] = out_ndarray

else:

    out_array[mask_ds.roi] = 255  # 255 means no myelin

# out_array[mask_ds.roi] = mask_ds.to_ndarray()
print("Writing myelin...")
out[mask_ds.roi] = out_array
