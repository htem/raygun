import daisy
import sys
import logging
# import numpy as np
import gt_tools

# from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
5/24/19:
- 

'''


if __name__ == "__main__":

    if sys.argv[1].endswith(".zarr"):
        out_file = sys.argv[1]
    else:
        config = gt_tools.load_config(sys.argv[1])
        out_file = config["out_file"]

    print(out_file)

    ds = daisy.open_ds(out_file, 'volumes/labels/labels_mask', 'r+')
    ds = daisy.open_ds(out_file, 'volumes/labels/labels_mask2', 'r+')
    ds[ds.roi] = 1
