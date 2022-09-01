import daisy
import sys
import logging
import numpy as np
# import gt_tools
import argparse

# from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    '''Example: python add_custom_labels_mask2.py /n/groups/htem/ESRF_id16a/jaspersLegCryo/groundTruthing/190322_r1_50nm_export2_labels_mask.zarr --roi_offset 2500 0 0 --roi_shape 9000 14000 14000'''

    ap = argparse.ArgumentParser(
        description=".")
    ap.add_argument("in_file", type=str, help='The input container') 
    ap.add_argument(
        "--roi_offset", type=int, help='',
        nargs='+', default=None)
    ap.add_argument(
        "--roi_shape", type=int, help='',
        nargs='+', default=None)

    # arg = sys.argv[1]
    args = ap.parse_args()

    # if arg.endswith(".zarr"):
    #     out_file = arg

    # else:
    #     config = gt_tools.load_config(sys.argv[1])
    #     out_file = config["out_file"]
    #     print(out_file)

    segment_ds = daisy.open_ds(args.in_file, 'volumes/labels/neuron_ids')

    roi = daisy.Roi(args.roi_offset, args.roi_shape)

    out = daisy.prepare_ds(
        args.in_file,
        "volumes/labels/labels_mask2",
        roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 5},
        delete=True
        )

    out[out.roi] = 1
