import daisy
import sys
import logging
import numpy as np
# import json
import os
import gt_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
'''

if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1])

    file = config["out_file"]

    raw = daisy.open_ds(file, "volumes/raw")

    roi = raw.roi

    if "effective_roi_context_nm" in config["CatmaidIn"]:
        roi_context = config["CatmaidIn"]["effective_roi_context_nm"]
    else:
        roi_context = config["CatmaidIn"]["roi_context_nm"]
    roi_context = daisy.Coordinate(tuple(roi_context))

    roi = roi.grow(-roi_context, -roi_context)

    if True:
        out_ds = daisy.prepare_ds(
            file,
            "volumes/labels/labels_mask2",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask2...")
        out_ds[out_ds.roi] = 1

    if True:
        out_ds = daisy.prepare_ds(
            file,
            "volumes/labels/labels_mask",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask...")
        out_ds[out_ds.roi] = 1

    if True:
        out_ds = daisy.prepare_ds(
            file,
            "volumes/labels/unlabeled",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing unlabeled...")
        out_ds[out_ds.roi] = 1

    if True:
        out_ds = daisy.prepare_ds(
            file,
            "volumes/labels/neuron_ids",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask...")
        out_ds[out_ds.roi] = 1
