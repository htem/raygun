import daisy
import sys
import logging
import numpy as np
# import json
# import os
import gt_tools
import time

from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Changelog:
'''

if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1])

    file = config["file"]
    out_file = config["out_file"]

    raw = daisy.open_ds(out_file, "volumes/raw")

    segment_ds_name = config["segment_ds_paintera_out"]
    # if "myelin_mask_file" in config:
    #     segment_ds_name = "volumes/labels/neuron_ids_myelin"
    # elif "skeleton_file" in config:
    #     #segment_ds_name = "volumes/segmentation_skeleton"
    #     segment_ds_name = config["segmentation_skeleton_ds"]

    segment_ds = daisy.open_ds(
        file,
        segment_ds_name)

    roi = segment_ds.roi

    background_ids = config["background_ids"]
    background_ids = [int(f) for f in background_ids]
    background_ids.append(0)
    ignored_fragments = config["ignored_fragments"]
    ignored_fragments = [int(f) for f in ignored_fragments]
    foreground_ids = config["foreground_ids"]
    foreground_ids = [int(f) for f in foreground_ids]

    print("Caching segment_ds...")
    t = time.time()
    segment_array = segment_ds[roi]
    segment_array.materialize()
    print("%s" % (time.time() - t))

    print("Making ignored fragments mask...")
    ignored_fragments_ndarray = np.ones(segment_array.shape, dtype=np.uint8)
    if len(ignored_fragments):
        print("Caching fragment_ds...")
        t = time.time()
        fragment_ds = daisy.open_ds(file, 'volumes/fragments')
        fragment_array = fragment_ds[roi]
        fragment_array.materialize()
        print("%s" % (time.time() - t))
        fragment_ndarray = fragment_array.to_ndarray()
        mask_values = ignored_fragments
        new_values = [0 for m in mask_values]
        replace_values(
            fragment_ndarray,
            mask_values,
            new_values,
            ignored_fragments_ndarray,
            )

    segment_ndarray = segment_array.to_ndarray()

    print("Making foreground mask...")
    foreground_mask = np.zeros_like(segment_ndarray, dtype=np.uint8)
    if len(foreground_ids):
        mask_values = foreground_ids
        new_values = [1 for m in foreground_ids]
        replace_values(
            segment_ndarray,
            mask_values,
            new_values,
            foreground_mask,
            )
    # print(foreground_mask)
    # print(foreground_ids); exit(0)

    if True:
        out_ds = daisy.prepare_ds(
            out_file,
            "volumes/labels/neuron_ids",
            roi,
            raw.voxel_size,
            segment_ds.dtype,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing neuron_ids...")
        # out_ndarray = np.ones_like(segment_ndarray, dtype=np.uint8)
        # out_ndarray = segment_ndarray
        # out_ds[out_ds.roi] = 1
        out_ndarray = np.copy(segment_ndarray)

        mask_values = background_ids
        new_values = [0 for m in mask_values]

        replace_values(
            segment_ndarray,
            mask_values,
            new_values,
            out_ndarray,
            )

        out_ds[roi] = out_ndarray

    if True:
        out_ds = daisy.prepare_ds(
            out_file,
            "volumes/labels/myelin_gt",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing myelin_gt...")
        out_ndarray = np.ones_like(segment_ndarray, dtype=np.uint8)
        mask_values = background_ids
        new_values = [0 for m in mask_values]
        replace_values(
            segment_ndarray,
            mask_values,
            new_values,
            out_ndarray,
            )
        myelin_ndarray = out_ndarray*255
        out_ds[out_ds.roi] = myelin_ndarray

    out_ndarray = np.logical_not(out_ndarray)
    out_ndarray = np.logical_and(out_ndarray, ignored_fragments_ndarray, dtype=np.uint8)
    out_ndarray = np.logical_or(out_ndarray, foreground_mask)

    if True:
        out_ds = daisy.prepare_ds(
            out_file,
            "volumes/labels/labels_mask2",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask2...")
        out_ds[out_ds.roi] = out_ndarray

    if True:
        out_ds = daisy.prepare_ds(
            out_file,
            "volumes/labels/labels_mask",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing labels_mask...")
        out_ds[out_ds.roi] = out_ndarray

    if True:
        out_ds = daisy.prepare_ds(
            out_file,
            "volumes/labels/unlabeled",
            roi,
            raw.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )
        print("Writing unlabeled...")
        out_ds[out_ds.roi] = out_ndarray
