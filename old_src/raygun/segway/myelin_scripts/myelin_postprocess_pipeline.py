# import json
# import os
import logging
import numpy as np
import daisy
from enum import Enum

from . import myelin_functions


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myelin_postprocess_pipeline")


def write_results(
        arrays,
        file,
        setupnum,
        write_roi,
        voxel_size,
        key,
        stepid,
        num_channels=1,
        ):

    data = arrays[key]

    output_ds = "debug/myelin_setup%02d/step%02d" % (setupnum, stepid)
    output_ds = daisy.prepare_ds(
        file,
        output_ds,
        write_roi,
        voxel_size,
        dtype=data.dtype,
        num_channels=num_channels,
        # write_roi=block_roi_z_dim,
        compressor={'id': 'zlib', 'level': 5})

    output_ds[output_ds.roi] = data


def run_postprocess_pipeline(
        block,
        # input_ds,
        file,
        setupnum,
        arrays,
        steps,
        voxel_size,
        # output_ds,
        ):

    '''TODO
    block has to have a read_roi and write roi so that we can run post processing in parallel
    or maybe it is not too necessary right now. ie, it may be better to run it for the whole volume
    '''

    # for each step, run step and save intermediate results for debugging
    for i, (fn_name, args) in enumerate(steps):

        # fn_name, args = step

        print("Running %s..." % (fn_name))

        if fn_name == "thresholding":
            fn = myelin_functions.thresholding

        elif fn_name == "make_affs_from_probability_map":
            fn = myelin_functions.affs_myelin_in_block

        elif fn_name == "extract_fragments":
            fn = myelin_functions.fragment_myelin_in_block

        elif fn_name == "agglomerate":
            fn = myelin_functions.agglomerate_myelin

        elif fn_name == "extract_segment":
            fn = myelin_functions.segment_myelin

        elif fn_name == "grow_segment_boundary":
            fn = myelin_functions.grow_segment_boundary

        elif fn_name == "clean_pred_with_segment_boundary":
            fn = myelin_functions.clean_pred_with_segment_boundary

        elif fn_name == "conv":
            fn = myelin_functions.conv

        elif fn_name == "make_affs_from_segments":
            fn = myelin_functions.make_affs_from_segments

        elif fn_name == "save":
            write_results(arrays, file, setupnum, block.read_roi, voxel_size, **args)
            continue

        else:
            assert False

        fn(arrays, block, **args)

        # print("Writing %s..." % (fn_name))
        # write_results(i, array, block.read_roi, input_ds.voxel_size)


if __name__ == "__main__":

    file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_04/cb2_synapse_cutout3_zoom4/cb2/130000/output.zarr"
    myelin_pred_ds = "volumes/myelin_affs"
    myelin_pred_ds = daisy.open_ds(file, myelin_pred_ds)
    setupnum = 0

    block = daisy.block.Block(myelin_pred_ds.roi, myelin_pred_ds.roi, myelin_pred_ds.roi)

    output_ds = "volumes/myelin_postprocess_setup00/myelin_affs_postprocessed"
    output_ds = daisy.prepare_ds(
        file,
        output_ds,
        myelin_pred_ds.roi,
        myelin_pred_ds.voxel_size,
        dtype=np.uint8,
        num_channels=3,
        compressor={'id': 'zlib', 'level': 5})

    class ArrayKeys(Enum):
        PRED = 1
        AFFS = 2
        RAG = 3
        FRAGMENTS = 4
        SEGMENT = 5
        PRED_CLEAN = 1

    steps = [
        ("thresholding", {
            "in_key": ArrayKeys.PRED,
            "out_key": ArrayKeys.PRED,
            "low_threshold": 128,
            }),
        ("save", {
            "key": ArrayKeys.PRED,
            "stepid": 0,
            }),

        ("make_affs_from_probability_map", {
            "in_key": ArrayKeys.PRED,
            "out_key": ArrayKeys.AFFS,
            "agglomerate_in_xy": False,
            }),
        ("save", {
            "key": ArrayKeys.AFFS,
            "stepid": 1,
            "num_channels": 3
            }),

        ("extract_fragments", {
            "in_key": ArrayKeys.AFFS,
            "out_fragment_key": ArrayKeys.FRAGMENTS,
            }),
        ("save", {
            "key": ArrayKeys.FRAGMENTS,
            "stepid": 2,
            }),

        ("agglomerate", {
            "affs_key": ArrayKeys.AFFS,
            "fragment_key": ArrayKeys.FRAGMENTS,
            "out_rag_key": ArrayKeys.RAG,
            "merge_function": "hist_quant_50",
            }),

        ("extract_segment", {
            "fragment_key": ArrayKeys.FRAGMENTS,
            "rag_key": ArrayKeys.RAG,
            "segment_key": ArrayKeys.SEGMENT,
            "threshold": .1,
            }),
        ("save", {
            "key": ArrayKeys.SEGMENT,
            "stepid": 3,
            }),

        ("grow_segment_boundary", {
            "segment_key": ArrayKeys.SEGMENT,
            "z_steps": 1,
            "xy_steps": 5,
            }),
        ("save", {
            "key": ArrayKeys.SEGMENT,
            "stepid": 4,
            }),

        ("clean_pred_with_segment_boundary", {
            "segment_key": ArrayKeys.SEGMENT,
            "pred_key": ArrayKeys.PRED,
            "pred_out_key": ArrayKeys.PRED_CLEAN
            }),
        ("save", {
            "key": ArrayKeys.PRED_CLEAN,
            "stepid": 5,
            }),

        ("conv", {
            "in_key": ArrayKeys.PRED_CLEAN,
            "out_key": ArrayKeys.PRED_CLEAN,
            "z_steps": 1,
            "xy_steps": 1,
            }),
        ("save", {
            "key": ArrayKeys.PRED_CLEAN,
            "stepid": 6,
            }),

        ]

    steps += [
        # run segmentation in 3D
        ("thresholding", {
            "in_key": ArrayKeys.PRED_CLEAN,
            "out_key": ArrayKeys.PRED_CLEAN,
            "low_threshold": 128,
            }),
        ("make_affs_from_probability_map", {
            "in_key": ArrayKeys.PRED_CLEAN,
            "out_key": ArrayKeys.AFFS,
            "agglomerate_in_xy": False,
            }),
        ("save", {
            "key": ArrayKeys.AFFS,
            "stepid": 7,
            "num_channels": 3
            }),
        ("extract_fragments", {
            "in_key": ArrayKeys.AFFS,
            "out_fragment_key": ArrayKeys.FRAGMENTS,
            }),
        ("save", {
            "key": ArrayKeys.FRAGMENTS,
            "stepid": 8,
            }),
        ("agglomerate", {
            "affs_key": ArrayKeys.AFFS,
            "fragment_key": ArrayKeys.FRAGMENTS,
            "out_rag_key": ArrayKeys.RAG,
            "merge_function": "hist_quant_75",
            }),
        ("extract_segment", {
            "fragment_key": ArrayKeys.FRAGMENTS,
            "rag_key": ArrayKeys.RAG,
            "segment_key": ArrayKeys.SEGMENT,
            "threshold": .80,
            }),
        ("save", {
            "key": ArrayKeys.SEGMENT,
            "stepid": 9,
            }),
        ("grow_segment_boundary", {
            "segment_key": ArrayKeys.SEGMENT,
            "z_steps": 0,
            "xy_steps": 2,
            }),
        ("save", {
            "key": ArrayKeys.SEGMENT,
            "stepid": 10,
            }),
        ]

    steps += [
        ("make_affs_from_segments", {
            "in_key": ArrayKeys.SEGMENT,
            "out_key": ArrayKeys.AFFS,
            }),
        ]

    # array = myelin_pred_ds[block.read_roi].to_ndarray()
    arrays = {
        ArrayKeys.PRED: myelin_pred_ds
    }

    run_postprocess_pipeline(block, file, setupnum, arrays, steps, myelin_pred_ds.voxel_size)

    print("Writing final results...")
    output_ds[output_ds.roi] = arrays[ArrayKeys.AFFS]


