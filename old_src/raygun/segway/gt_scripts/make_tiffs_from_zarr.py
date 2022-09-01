import daisy
from daisy import Coordinate, Roi
import numpy as np
from PIL import Image
import sys
import json
import os

config_f = sys.argv[1]
with open(config_f) as f:
    config = json.load(f)
script_name = os.path.basename(config_f)
script_name = script_name.split(".")[0]

cutout_ds = daisy.open_ds(config["raw_file"], config["raw_ds"])
# try:
#     cutout_ds = daisy.open_ds(config["raw_file"], config["raw_ds"])
# except:
#     path = os.path.join(
#         os.path.split(config_f)[0],
#         config["zarr"]["dir"],
#         script_name + ".zarr")
#     print(path)
#     cutout_ds = daisy.open_ds(
#         path, "volumes/raw")

voxel_size = cutout_ds.voxel_size
coord_begin = Coordinate(np.flip(np.array(config['coord_begin']))) * voxel_size
coord_end = Coordinate(np.flip(np.array(config['coord_end']))) * voxel_size

roi_offset = coord_begin
roi_shape = coord_end - coord_begin
roi = Roi(roi_offset, roi_shape)

print(f'Making GT with ROI (in ZYX nm) {roi}...')

os.makedirs(script_name, exist_ok=True)
output_dir = script_name

print("Getting data from zarr file...")
ndarray = cutout_ds[roi].to_ndarray()

print("Writing tiffs...")
z_len = ndarray.shape[0]
for section_num in range(z_len):
    fpath = f'{output_dir}/{section_num}.tiff'
    print(fpath)
    section = ndarray[section_num]
    tile = Image.fromarray(section)
    tile.save(fpath, quality=95)
