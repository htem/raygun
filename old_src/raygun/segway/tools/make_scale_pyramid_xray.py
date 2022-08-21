import os
import sys
import re
sys.path.append("/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/scale_pyramid")

from scale_pyramid import create_scale_pyramid

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    in_path = sys.argv[1].rstrip('/')
    in_file = in_path.split(".zarr/")[0] + '.zarr'
    in_ds = in_path.split(".zarr/")[1]

    # if the s0 mipmap has not been created, make it
    make_symlink = True
    if re.match("s\d+", in_path.split('/')[-1]):
        # skip if already given s0, s1, s2, ...
        make_symlink = False

    elif os.path.exists(os.path.join(in_path, "s0")):
        # check if ds already has s0 created
        make_symlink = False
        in_ds = os.path.join(in_ds, "s0")

    if make_symlink:
        in_path_mipmap = in_path + "_mipmap"
        os.makedirs(in_path_mipmap)
        os.symlink(os.path.realpath(in_path), os.path.join(in_path_mipmap, "s0"))
        in_ds = os.path.join(in_ds + "_mipmap", "s0")
        print("Making new mipmap ds: ", in_ds)

    scales = [
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        ]

    chunk_shape = [128, 128, 128]

    create_scale_pyramid(
        in_file,
        in_ds,
        scales,
        chunk_shape)
