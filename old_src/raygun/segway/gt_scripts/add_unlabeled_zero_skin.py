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

    roi = segment_ds.roi
    roi_begin = [k for k in roi.get_begin()]
    roi_begin[0] = 200*16
    roi_begin[2] = 1000*16
    roi_shape = [k for k in roi.get_shape()]
    roi_shape[0] = 600*16
    roi_shape[2] = 1200*16
    roi = daisy.Roi(roi_begin, roi_shape)

    out_ds = daisy.prepare_ds(
        out_file,
        "volumes/labels/unlabeled",
        roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 5},
        delete=True
        )

    unlabeled_ndarray = np.ones(out_ds.shape, dtype=out_ds.dtype)

    labels_ndarray = segment_ds[roi].to_ndarray()
    segment_by_foreground = [0]
    new_mask_values = [0]
    replace_values(
        labels_ndarray,
        segment_by_foreground,
        new_mask_values,
        unlabeled_ndarray)

    out_ds[out_ds.roi] = unlabeled_ndarray
