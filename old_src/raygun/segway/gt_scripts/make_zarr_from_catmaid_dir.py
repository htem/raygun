import daisy
from daisy import Coordinate
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json
import os

import gt_tools

config_f = sys.argv[1]
# with open(config_f) as f:
#     config = json.load(f)

config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)

script_name = os.path.basename(config_f)
script_name = script_name.split(".")[0]

# path to raw
in_config = config["CatmaidIn"]
raw_f = in_config["file"]
voxel_size = in_config["voxel_size"]
tile_shape = in_config["tile_shape"]
raw_dir_shape = [m*n for m, n in zip(tile_shape, voxel_size)]  # zyx in nm

roi_offset = in_config["roi_offset"]
if in_config["roi_offset_encoding"] == "tile":
    roi_offset = [m*n for m, n in zip(roi_offset, raw_dir_shape)]
else:
    raise RuntimeError("Unknown offset encoding")

additional_offset = in_config.get("roi_additional_offset_nm", [0, 0, 0])

roi_context = in_config.get("roi_context_nm", [0, 0, 0])

roi_shape = in_config["roi_shape_nm"]

leave_blank_if_missing = in_config.get("leave_blank_if_missing", False)

out_config = config["zarr"]
cutout_f = out_config["dir"] + "/" + script_name + ".zarr"
print(cutout_f)
cutout_f = config["out_file"]
print(cutout_f)
# exit()
cutout_f_with_context = out_config["dir"] + "/" + script_name + "with_context.zarr"
zero_offset = out_config.get("zero_offset", True)
xy_downsample = out_config.get("xy_downsample", 1)

write_voxel_size = voxel_size
if xy_downsample > 1:
    write_voxel_size = []
    write_voxel_size.append(voxel_size[0])
    write_voxel_size.append(voxel_size[1] * xy_downsample)
    write_voxel_size.append(voxel_size[2] * xy_downsample)
    write_voxel_size = tuple(write_voxel_size)

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
roi_offset[0] += additional_offset[0]
roi_offset[1] += additional_offset[1]
roi_offset[2] += additional_offset[2]
roi_shape[0] += roi_context[0]
roi_shape[1] += roi_context[1]
roi_shape[2] += roi_context[2]
roi_shape[0] += roi_context[0]
roi_shape[1] += roi_context[1]
roi_shape[2] += roi_context[2]

write_roi_abs = daisy.Roi(roi_offset, roi_shape)

replace_section_list = {}
if 'replace_section_list' in in_config:
    for pair in in_config['replace_section_list']:
        replace_section_list[pair[0]] = pair[1]

cutout_ds = daisy.prepare_ds(
    cutout_f,
    'volumes/raw',
    write_roi,
    write_voxel_size,
    np.uint8,
    compressor=None,
    delete=True
    )

cutout_nd = np.zeros(write_roi.get_shape()/daisy.Coordinate(write_voxel_size), dtype=np.uint8)
cutout_array = daisy.Array(cutout_nd, write_roi, write_voxel_size)

# exit(0)

# cutout_with_context_ds = daisy.prepare_ds(
#     cutout_f_with_context,
#     'raw',
#     write_roi_with_context,
#     write_voxel_size,
#     np.uint8,
#     # write_size=raw_dir_shape,
#     # num_channels=self.out_dims,
#     # temporary fix until
#     # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
#     # (we want gzip to be the default)
#     compressor={'id': 'zlib', 'level': 5}
#     )


print("write_roi: %s" % write_roi)
print("raw_dir_shape: %s" % raw_dir_shape)

# for i in range(len(voxel_size)):
#     assert((raw_dir_shape[i] % voxel_size[i]) == 0)
#     assert((roi_offset[i] % voxel_size[i]) == 0)
#     assert((roi_offset[i] % raw_dir_shape[i]) == 0)
#     assert((roi_shape[i] % voxel_size[i]) == 0)

# # z-direction needs to be aligned
# assert((roi_shape[0] % raw_dir_shape[0]) == 0)

# print("ROI ")

z_begin = int(roi_offset[0] / raw_dir_shape[0])
z_end = int((roi_offset[0] + roi_shape[0]) / raw_dir_shape[0])
y_begin = int(roi_offset[1] / raw_dir_shape[1])
y_end = int((roi_offset[1] + roi_shape[1]) / raw_dir_shape[1])
if (roi_offset[1] + roi_shape[1]) % raw_dir_shape[1]:
    y_end += 1
x_begin = int(roi_offset[2] / raw_dir_shape[2])
x_end = int((roi_offset[2] + roi_shape[2]) / raw_dir_shape[2])
if (roi_offset[2] + roi_shape[2]) % raw_dir_shape[2]:
    x_end += 1

tile_size = None

for z_index in range(z_begin, z_end):
    for y_index in range(y_begin, y_end):
        for x_index in range(x_begin, x_end):

            found = False
            z_index_tmp = z_index
            while not found:

                if z_index_tmp in replace_section_list:
                    z_index_tmp = replace_section_list[z_index_tmp]

                tile_f = raw_f + ('/%d/%d/%d.jpg' % (z_index_tmp, y_index, x_index))
                try:
                    tile = Image.open(tile_f)
                except:
                    print("Missing %s" % tile_f)
                    if not leave_blank_if_missing:
                        z_index_tmp -= 1
                        # if z_index_tmp not in replace_section_list:
                        #     raise RuntimeError("%s not found and not in replace_section_list" % tile_f)
                        # exit(0)
                        continue
                    else:
                        tile = Image.new('L', tile_size)
                        print("Blank %s" % tile_f)
                        # exit(0)

                tile_roi = daisy.Roi(
                    (
        			z_index*voxel_size[0]*tile_shape[0],
        			y_index*voxel_size[1]*tile_shape[1],
        			x_index*voxel_size[2]*tile_shape[2],
        		    ),
                    (raw_dir_shape[0], raw_dir_shape[1], raw_dir_shape[2]))

                if tile_size is None:
                    tile_size = tile.size

                # print("tile_roi:", tile_roi)
                # print("write_roi_abs:", write_roi_abs)
                local_write_roi_abs = write_roi_abs.intersect(tile_roi)
                # print("local_write_roi_abs: %s" % local_write_roi_abs)

                crop_x_begin = local_write_roi_abs.get_begin()[2] % (voxel_size[2]*tile_shape[2])
                crop_y_begin = local_write_roi_abs.get_begin()[1] % (voxel_size[1]*tile_shape[1])
                crop_x_end = crop_x_begin + local_write_roi_abs.get_shape()[2]
                crop_y_end = crop_y_begin + local_write_roi_abs.get_shape()[1]
                # print("crop_x_begin: %s" % crop_x_begin)
                # print("crop_y_begin: %s" % crop_y_begin)
                # print("crop_x_end: %s" % crop_x_end)
                # print("crop_y_end: %s" % crop_y_end)

                tile = tile.crop((
                                 int(crop_x_begin / voxel_size[2]),
                                 int(crop_y_begin / voxel_size[1]),
                                 int(crop_x_end / voxel_size[2]),
                                 int(crop_y_end / voxel_size[1])))

                tile_array = np.asarray(tile).reshape(
                    1,
                    int(local_write_roi_abs.get_shape()[1]/voxel_size[1]/xy_downsample),
                    int(local_write_roi_abs.get_shape()[2]/voxel_size[2]/xy_downsample))

                if not leave_blank_if_missing:
                    if len(replace_section_list) == 0 and np.sum(tile_array) == 0:
                        print("Blacked out %s" % tile_f)
                        z_index_tmp -= 1
                        continue

                found = True

            print(tile_f)
            # print(tile_array)
            # print(slice_roi_shape)

            # slice_roi_offset = (z_index*raw_dir_shape[0],
            #                     y_index*raw_dir_shape[1],
            #                     x_index*raw_dir_shape[2])
            # print(slice_roi_offset)
            local_write_roi = local_write_roi_abs
            if zero_offset:
                local_write_roi -= Coordinate(roi_offset)
            print(local_write_roi)

            cutout_array[local_write_roi] = tile_array

print("Writing to disk...")
cutout_ds[cutout_ds.roi] = cutout_array
