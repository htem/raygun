import daisy
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json
import os

import gt_tools

config_f = sys.argv[1]
# with open(config_f) as f:
    # config = json.load(f)
config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)
script_name = os.path.basename(config_f)
script_name = script_name.split(".")[0]

# path to raw
in_config = config["CatmaidIn"]
raw_f = in_config["file"]
voxel_size = in_config["voxel_size"]
raw_dir_shape = [m*n for m, n in zip(in_config["tile_shape"], voxel_size)]  # zyx in nm
bad_slices = in_config["bad_slices"]
# print(bad_slices); exit(0)

roi_offset = in_config["roi_offset"]
if in_config["roi_offset_encoding"] == "tile":
    roi_offset = [m*n for m, n in zip(roi_offset, raw_dir_shape)]
else:
    raise RuntimeError("Unknown offset encoding")

additional_offset = in_config.get("roi_additional_offset_nm", [0, 0, 0])

roi_context = in_config.get("roi_context_nm", [0, 0, 0])

roi_shape = in_config["roi_shape_nm"]

out_config = config["zarr"]
# cutout_f = out_config["dir"] + "/" + script_name + ".zarr"
cutout_f = config["out_file"]
zero_offset = out_config.get("zero_offset", True)
xy_downsample = out_config.get("xy_downsample", 1)

# in nm
# roi_offset = (78*40, 96*1024*4, 125*1024*4)
# roi_shape = (4000, 4096, 4096)
# roi_offset = (175*40, 96*1024*4, 125*1024*4)
# roi_shape = (80, 4096, 4096)

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
    daisy.Coordinate([0, 0, 0]), daisy.Coordinate(roi_context))
write_roi_with_context = write_roi_with_context.grow(
    daisy.Coordinate([0, 0, 0]), daisy.Coordinate(roi_context))
print(write_roi_with_context)
write_roi = write_roi_with_context

# roi_offset -= roi_context
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
# exit(0)

cutout_ds = daisy.prepare_ds(
    cutout_f,
    'volumes/raw',
    write_roi,
    write_voxel_size,
    np.uint8,
    compressor=None
    )

# exit(0)



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

bad_slices = [n + z_begin for n in bad_slices]

# print(bad_slices); exit(0)

for z_index in bad_slices:
    for y_index in range(y_begin, y_end):
        for x_index in range(x_begin, x_end):

            found = False
            print("bad slice: ", z_index)
            z_index_tmp = z_index - 1  # get the previous slice
            while not found:

                if z_index_tmp in bad_slices:
                    z_index_tmp -= 1
                    continue

                tile_f = raw_f + ('/%d/%d/%d.jpg' % (z_index_tmp, y_index, x_index))
                try:
                    tile = Image.open(tile_f)
                except:
                    print("Missing %s" % tile_f)
                    z_index_tmp -= 1
                    #exit(0)
                    continue

                tile_roi = daisy.Roi(
                    (z_index*40, y_index*4*1024, x_index*4*1024),
                    (raw_dir_shape[0], raw_dir_shape[1], raw_dir_shape[2]))

                print("tile_roi: ", tile_roi)
                print("write_roi_abs: ", write_roi_abs)
                local_write_roi_abs = write_roi_abs.intersect(tile_roi)
                print("local_write_roi_abs: %s" % local_write_roi_abs)
                # exit(0)

                # slice_roi_shape = raw_dir_shape.copy()
                # # print(x_index)
                # # print(x_end)
                # # print(roi_shape[2] % raw_dir_shape[2])
                # # print(slice_roi_shape)
                # x_begin = 0
                # y_begin = 0
                # x_end = raw_dir_shape[2]
                # y_end = raw_dir_shape[1]

                # if x_index == (x_end-1) and (roi_shape[2] % raw_dir_shape[2]):
                #     x_end = roi_shape[2] % raw_dir_shape[2]
                #     slice_roi_shape[2] = x_end
                # if y_index == (y_end-1) and (roi_shape[1] % raw_dir_shape[1]):
                #     y_end = roi_shape[1] % raw_dir_shape[1]
                #     slice_roi_shape[1] = y_end
                # # print(slice_roi_shape)
                crop_x_begin = local_write_roi_abs.get_begin()[2] % (4*1024)
                crop_y_begin = local_write_roi_abs.get_begin()[1] % (4*1024)
                crop_x_end = crop_x_begin + local_write_roi_abs.get_shape()[2]
                crop_y_end = crop_y_begin + local_write_roi_abs.get_shape()[1]
                print("crop_x_begin: %s" % crop_x_begin)
                print("crop_y_begin: %s" % crop_y_begin)
                print("crop_x_end: %s" % crop_x_end)
                print("crop_y_end: %s" % crop_y_end)
                # # exit(0)
                # tile = tile.crop((2048, 2048, 4096, 2048 + 2))
                # print(np.asarray(tile))
                # exit(0)


                tile = tile.crop((
                                 int(crop_x_begin / voxel_size[2]),
                                 int(crop_y_begin / voxel_size[1]),
                                 int(crop_x_end / voxel_size[2]),
                                 int(crop_y_end / voxel_size[1])))

                tile_array = np.asarray(tile).reshape(
                    1,
                    int(local_write_roi_abs.get_shape()[1]/voxel_size[1]/xy_downsample),
                    int(local_write_roi_abs.get_shape()[2]/voxel_size[2]/xy_downsample))

                if np.sum(tile_array) == 0:
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
                local_write_roi -= daisy.Coordinate(roi_offset)
            # print("local_write_roi_abs: ", local_write_roi_abs)
            # print("roi_offset: ", roi_offset)
            # print("local_write_roi: ", local_write_roi)
            # exit(0)

            # print(slice_roi_offset)
            # slice_roi = daisy.Roi(slice_roi_offset, slice_roi_shape)
            # print("slice_roi: %s" % slice_roi)
            # print("cutout_ds.roi: %s" % cutout_ds.roi)
            # print(cutout_ds.roi)
            # print(np.asarray(tile))
            # exit(0)
            cutout_ds[local_write_roi] = tile_array
            # cutout_ds[local_write_roi] = 0
