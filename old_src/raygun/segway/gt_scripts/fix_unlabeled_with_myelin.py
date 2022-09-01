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

    unlabeled = daisy.open_ds(file, "volumes/labels/unlabeled", mode='r+')
    myelin = daisy.open_ds(file, "volumes/labels/myelin_gt")

    myelin_ndarray = myelin.to_ndarray()
    unlabeled_ndarray = unlabeled.to_ndarray()
    np.place(unlabeled_ndarray, myelin_ndarray == 0, 0)
    unlabeled[unlabeled.roi] = unlabeled_ndarray

    if True:
        out_ds = daisy.prepare_ds(
            file,
            "volumes/labels/labels_mask2",
            unlabeled.roi,
            unlabeled.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask2...")
        out_ds[out_ds.roi] = 1

