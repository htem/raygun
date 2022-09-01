# TODO: make actually use percent hold out
#adapted from code written by Tri Nguyen (Harvard, 2021)
import daisy
import sys
import logging
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("fpath", type=str, help='')
    ap.add_argument("--percent_val", type=float, default=0.1, help='fraction of data to hold out for validation', nargs='+')
    ap.add_argument("--unfinished_sections", type=int, default=None, help='', nargs='+')
    ap.add_argument("--finished_sections", type=int, default=None, help='', nargs='+')
    ap.add_argument("--roi_offset", type=int, default=None, help='', nargs='+')
    ap.add_argument("--roi_shape", type=int, default=None, help='', nargs='+')
    ap.add_argument("--pixel_coords", type=int, default=1, help='')
    ap.add_argument("--src_ds", type=str, default="volumes/raw", help='')
    ap.add_argument("--output_ds", type=str, default="volumes/masks/validation_mask", help='')
    ap.add_argument("--whole_roi", type=int, default=0, help='')
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v
    # arg = sys.argv[1]


    assert finished_sections is None or unfinished_sections is None, "Only one can be specified"

    # print(unfinished_sections); exit()

    if fpath.endswith(".zarr") or fpath.endswith(".n5"):
        out_file = fpath

    segment_ds = daisy.open_ds(out_file, src_ds)

    roi = None
    if roi_offset or roi_shape:
        roi = daisy.Roi(roi_offset, roi_shape)
        if pixel_coords:
            roi *= segment_ds.voxel_size

    if roi is None:
        roi = segment_ds.roi

    print(roi)
    ds_roi = roi
    if whole_roi:
        ds_roi = segment_ds.roi

    out = daisy.prepare_ds(
        out_file,
        output_ds,
        ds_roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 3},
        delete=True
        )
    # out[segment_ds.roi] = 0
    #TODO: make mask with percent_val held out
    out[roi] = 1
    exit()

    arr_shape = roi.get_shape() / segment_ds.voxel_size

    if finished_sections is not None:
        arr = np.zeros(arr_shape, dtype=out.dtype)
        for s in finished_sections:
            arr[s, :, :] = 1

    else:
        arr = np.ones(arr_shape, dtype=out.dtype)
        if unfinished_sections is not None:
            for s in unfinished_sections:
                arr[s, :, :] = 0

    out[out.roi] = arr
