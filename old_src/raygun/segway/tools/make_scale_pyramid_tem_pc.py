
import sys
import daisy
sys.path.append("/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/scale_pyramid")

from scale_pyramid import create_scale_pyramid

if __name__ == "__main__":

    in_file = sys.argv[1].split(".zarr/")[0] + '.zarr'
    in_ds = sys.argv[1].split(".zarr/")[1]

    scales = [
        (1, 2, 2),
        (1, 2, 2),
        (1, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        ]

    chunk_shape = [50, 128, 128]

    roi = daisy.Roi(
        tuple([4000, 243712, 325632]),
        tuple([16000, 65536, 65536]))

    roi = daisy.Roi(
        tuple([4000, 88064, 108544]),
        tuple([16000, 348160, 348160]))

    create_scale_pyramid(
        in_file,
        in_ds,
        scales,
        chunk_shape,
        roi=roi)
