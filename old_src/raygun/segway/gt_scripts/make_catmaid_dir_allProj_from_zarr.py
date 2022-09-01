import daisy
from daisy import Coordinate, Roi
# import neuroglancer
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

xy_downsample = 1

try:
    cutout_ds = daisy.open_ds(config["raw_file"], config["raw_ds"])
except:
    path = os.path.join(
        os.path.split(config_f)[0],
        config["zarr"]["dir"],
        script_name + ".zarr")
    print(path)
    cutout_ds = daisy.open_ds(
        path, "volumes/raw")

voxel_size = cutout_ds.voxel_size

catmaid_folder = config["CatmaidOut"].get("folder", script_name)
catmaid_f = config["CatmaidOut"]["dir"] + "/" + catmaid_folder
catmaid_f_xy = config["CatmaidOut"]["dir"] + "/" + catmaid_folder + "/xy"
catmaid_f_xz = config["CatmaidOut"]["dir"] + "/" + catmaid_folder + "/xz"
catmaid_f_zy = config["CatmaidOut"]["dir"] + "/" + catmaid_folder + "/zy"

roi_offset = cutout_ds.roi.get_begin()
print("offset = %s" % str(roi_offset))
roi_shape = cutout_ds.roi.get_end() - roi_offset
print("shape = %s" % str(roi_shape))

# do xy proj
raw_dir_shape = [voxel_size[0], roi_shape[1], roi_shape[2]]
print(raw_dir_shape)

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

for z_index in range(z_begin, z_end):
    for y_index in range(y_begin, y_end):
        for x_index in range(x_begin, x_end):

            os.makedirs(catmaid_f_xy + '/0/%d/%d' % (z_index, y_index), exist_ok=True)
            fpath = catmaid_f_xy + '/0/%d/%d/%d.jpg' % (z_index, y_index, x_index)

            slice_roi_shape = raw_dir_shape.copy()
            x_len = raw_dir_shape[2]
            y_len = raw_dir_shape[1]
            if x_index == (x_end-1) and (roi_shape[2] % raw_dir_shape[2]):
                x_len = roi_shape[2] % raw_dir_shape[2]
                slice_roi_shape[2] = x_len
            if y_index == (y_end-1) and (roi_shape[1] % raw_dir_shape[1]):
                y_len = roi_shape[1] % raw_dir_shape[1]
                slice_roi_shape[1] = y_len

            slice_roi_offset = (z_index*raw_dir_shape[0],
                                y_index*raw_dir_shape[1],
                                x_index*raw_dir_shape[2])
            slice_roi = daisy.Roi(slice_roi_offset, slice_roi_shape)

            print(cutout_ds.roi)
            print(slice_roi)
            slice_array = cutout_ds[slice_roi].to_ndarray().reshape(
                    int(slice_roi_shape[1]/voxel_size[1]/xy_downsample),
                    int(slice_roi_shape[2]/voxel_size[2]/xy_downsample))

            print(fpath)
            tile = Image.fromarray(slice_array)
            tile.save(fpath, quality=95)

            continue



# do xz proj
raw_dir_shape = [roi_shape[0], voxel_size[0], roi_shape[2]]
print(raw_dir_shape)

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

for y_index in range(y_begin, y_end):
    for z_index in range(z_begin, z_end):
        for x_index in range(x_begin, x_end):

            os.makedirs(catmaid_f_xz + '/0/%d/%d' % (y_index, z_index), exist_ok=True)
            fpath = catmaid_f_xz + '/0/%d/%d/%d.jpg' % (y_index, z_index, x_index)

            slice_roi_shape = raw_dir_shape.copy()
            x_len = raw_dir_shape[2]
            z_len = raw_dir_shape[0]
            if x_index == (x_end-1) and (roi_shape[2] % raw_dir_shape[2]):
                x_len = roi_shape[2] % raw_dir_shape[2]
                slice_roi_shape[2] = x_len
            if z_index == (z_end-1) and (roi_shape[0] % raw_dir_shape[0]):
                z_len = roi_shape[0] % raw_dir_shape[0]
                slice_roi_shape[0] = z_len

            slice_roi_offset = (z_index*raw_dir_shape[0],
                                y_index*raw_dir_shape[1],
                                x_index*raw_dir_shape[2])
            slice_roi = daisy.Roi(slice_roi_offset, slice_roi_shape)

            print(cutout_ds.roi)
            print(slice_roi)
            slice_array = cutout_ds[slice_roi].to_ndarray().reshape(
                    int(slice_roi_shape[0]/voxel_size[0]/xy_downsample),
                    int(slice_roi_shape[2]/voxel_size[2]/xy_downsample))

            print(fpath)
            tile = Image.fromarray(slice_array)
            tile.save(fpath, quality=95)

            continue

# do yz proj
raw_dir_shape = [roi_shape[0], roi_shape[1],voxel_size[0]]
print(raw_dir_shape)

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

for x_index in range(x_begin, x_end):
    for y_index in range(y_begin, y_end):
        for z_index in range(z_begin, z_end):


            os.makedirs(catmaid_f_zy + '/0/%d/%d' % (x_index, y_index), exist_ok=True)
            fpath = catmaid_f_zy + '/0/%d/%d/%d.jpg' % (x_index, y_index, z_index)

            slice_roi_shape = raw_dir_shape.copy()
            y_len = raw_dir_shape[1]
            z_len = raw_dir_shape[0]
            if y_index == (x_end-1) and (roi_shape[1] % raw_dir_shape[1]):
                y_len = roi_shape[1] % raw_dir_shape[1]
                slice_roi_shape[1] = y_len
            if z_index == (z_end-1) and (roi_shape[0] % raw_dir_shape[0]):
                z_len = roi_shape[0] % raw_dir_shape[0]
                slice_roi_shape[0] = z_len

            slice_roi_offset = (z_index*raw_dir_shape[0],
                                y_index*raw_dir_shape[1],
                                x_index*raw_dir_shape[2])
            slice_roi = daisy.Roi(slice_roi_offset, slice_roi_shape)

            print(cutout_ds.roi)
            print(slice_roi)
            slice_array = cutout_ds[slice_roi].to_ndarray().reshape(
                    int(slice_roi_shape[1]/voxel_size[1]/xy_downsample),
                    int(slice_roi_shape[0]/voxel_size[0]/xy_downsample))

            print(fpath)
            tile = Image.fromarray(np.transpose(slice_array))
            tile.save(fpath, quality=95)

            continue
