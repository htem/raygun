# import json
# import os
import logging
# import numpy as np
# import daisy
from enum import Enum

from . import myelin_postprocess_pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myelin_postprocess_pipeline")


def run_postprocess_setup(
        block,
        file,
        myelin_pred_ds,
        myelin_affs,
        ):

    # myelin_affs_ds = daisy.prepare_ds(
    #     file,
    #     myelin_affs_ds,
    #     myelin_pred_ds.roi,
    #     myelin_pred_ds.voxel_size,
    #     dtype=np.uint8,
    #     num_channels=3,
    #     compressor={'id': 'zlib', 'level': 5})

    class ArrayKeys(Enum):
        PRED = 1
        AFFS = 2
        RAG = 3
        FRAGMENTS = 4
        SEGMENT = 5
        PRED_CLEAN = 6

    setupnum = 0

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
        ("save", {
            "key": ArrayKeys.AFFS,
            "stepid": 11,
            "num_channels": 3
            }),
        ]

    # array = myelin_pred_ds[block.read_roi].to_ndarray()
    arrays = {
        ArrayKeys.PRED: myelin_pred_ds
    }

    myelin_postprocess_pipeline.run_postprocess_pipeline(
        block, file, setupnum, arrays, steps, myelin_pred_ds.voxel_size)

    # print("Writing final results...")
    # myelin_affs_ds[block.write_roi] = arrays[ArrayKeys.AFFS][block.write_roi].to_ndarray()

    # return arrays[ArrayKeys.AFFS]
    myelin_affs[block.write_roi] = arrays[ArrayKeys.AFFS].to_ndarray()
