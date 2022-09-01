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

    gt = daisy.open_ds(file, "volumes/labels/neuron_ids")

    if True:
        out_ds = daisy.prepare_ds(
            file,
            "volumes/labels/labels_mask2",
            gt.roi,
            gt.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask2...")
        out_ds[out_ds.roi] = 1

