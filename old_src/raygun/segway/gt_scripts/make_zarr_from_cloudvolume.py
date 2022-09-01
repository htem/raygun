import daisy
from daisy import Coordinate
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json
import os

from cloudvolume import CloudVolume, view

import gt_tools

config_f = sys.argv[1]
# with open(config_f) as f:
#     config = json.load(f)

config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)

script_name = os.path.basename(config_f)
script_name = script_name.split(".")[0]

# path to raw
in_config = config["CloudVolumeIn"]
url = in_config["url"]
voxel_size = in_config["voxel_size"]
mip0_voxel_size = in_config["mip0_voxel_size"]
mip = in_config["mip"]
transpose_source = in_config["transpose_source"]

roi_offset = in_config.get("roi_offset", [0, 0, 0])

roi_context = in_config.get("roi_context_nm", [0, 0, 0])

roi_shape = in_config["roi_shape_nm"]

# leave_blank_if_missing = in_config.get("leave_blank_if_missing", False)

out_config = config["zarr"]
out_dir = out_config.get("dir", '.')
cutout_f = out_dir + "/" + script_name + ".zarr"
# print(cutout_f)
cutout_f = config.get("out_file", cutout_f)
print(cutout_f)
# exit()
# cutout_f_with_context = out_config["dir"] + "/" + script_name + "with_context.zarr"
zero_offset = out_config.get("zero_offset", True)
xy_downsample = out_config.get("xy_downsample", 1)

write_voxel_size = voxel_size
if xy_downsample > 1:
    write_voxel_size = []
    write_voxel_size.append(voxel_size[0])
    write_voxel_size.append(voxel_size[1] * xy_downsample)
    write_voxel_size.append(voxel_size[2] * xy_downsample)
    write_voxel_size = tuple(write_voxel_size)
write_voxel_size = Coordinate(write_voxel_size)

write_roi = daisy.Roi(roi_offset, roi_shape)
write_roi_abs = write_roi
if zero_offset:
    write_roi = daisy.Roi((0, 0, 0), roi_shape)  # 0,0,0 offset because we already traced

print(write_roi)
write_roi_with_context = write_roi.grow(
    Coordinate([0, 0, 0]), Coordinate(roi_context))
write_roi_with_context = write_roi_with_context.grow(
    Coordinate([0, 0, 0]), Coordinate(roi_context))
print(write_roi_with_context)
write_roi = write_roi_with_context

roi_offset[0] -= roi_context[0]
roi_offset[1] -= roi_context[1]
roi_offset[2] -= roi_context[2]
# roi_offset[0] += additional_offset[0]
# roi_offset[1] += additional_offset[1]
# roi_offset[2] += additional_offset[2]
roi_shape[0] += roi_context[0]
roi_shape[1] += roi_context[1]
roi_shape[2] += roi_context[2]
roi_shape[0] += roi_context[0]
roi_shape[1] += roi_context[1]
roi_shape[2] += roi_context[2]

# write_roi_abs = daisy.Roi(roi_offset, roi_shape)

replace_section_list = {}
if 'replace_section_list' in in_config:
    for pair in in_config['replace_section_list']:
        replace_section_list[pair[0]] = pair[1]

chunk_size = Coordinate(config['zarr'].get('chunk_size', None))*write_voxel_size
zarr_ds = config.get('raw_dataset', 'volumes/raw')
cutout_ds = daisy.prepare_ds(
    cutout_f,
    zarr_ds,
    write_roi,
    write_voxel_size,
    np.uint8,
    compressor={'id': 'zlib', 'level': 1},
    delete=True,
    write_size=chunk_size
    )

# cutout_nd = np.zeros(write_roi.get_shape()/daisy.Coordinate(write_voxel_size), dtype=np.uint8)
# cutout_array = daisy.Array(cutout_nd, write_roi, write_voxel_size)

voxel_coord = [k for k in Coordinate(roi_offset) / Coordinate(mip0_voxel_size)]
size_in_voxel = [k for k in Coordinate(roi_shape) / Coordinate(voxel_size)]

print("voxel_coord:", voxel_coord)
print("size_in_voxel:", size_in_voxel)

for x in size_in_voxel:
    if x % 2:
        assert False, f'size is not even {size_in_voxel}'

# print(f'{write_roi.get_begin()/2}')
centered_voxel_coord = write_roi.get_shape()/2/Coordinate(mip0_voxel_size) + Coordinate(voxel_coord)
print("centered_voxel_coord:", centered_voxel_coord)

if transpose_source:
    centered_voxel_coord = np.flip(centered_voxel_coord)
    size_in_voxel = np.flip(size_in_voxel)

cv = CloudVolume(
    url,
    progress=True,
    use_https=True,
    max_redirects=0,
    )

img = cv.download_point(centered_voxel_coord, mip=mip, size=size_in_voxel)

if transpose_source:
    img = img.transpose()

if len(img.shape) == 4:
    img = img[0, :, :, :]
else:
    assert len(img.shape) == 3

print("Writing to disk...")
cutout_ds[cutout_ds.roi] = img
