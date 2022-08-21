import daisy
import sys
import logging
import numpy as np
import gt_tools

from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
5/24/19:
- 

'''


if __name__ == "__main__":

    arg = sys.argv[1]

    if arg.endswith(".zarr"):
        out_file = arg

    else:
        config = gt_tools.load_config(sys.argv[1])
        out_file = config["out_file"]
        print(out_file)

    segment_ds = daisy.open_ds(out_file, 'volumes/labels/neuron_ids')

    out_ds = daisy.prepare_ds(
        out_file,
        "volumes/labels/unlabeled",
        segment_ds.roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 5},
        delete=True
        )

    unlabeled_ndarray = np.ones(out_ds.shape, dtype=out_ds.dtype)

    labels_ndarray = segment_ds.to_ndarray()
    segment_by_foreground = [0]
    new_mask_values = [0]
    replace_values(
        labels_ndarray,
        segment_by_foreground,
        new_mask_values,
        unlabeled_ndarray)

    out_ds[out_ds.roi] = unlabeled_ndarray
