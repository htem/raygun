import daisy
# from daisy import Coordinate
import numpy as np
import sys
import json
import os
import logging
import gt_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("make_zarr_from_cb2_v2_zarr")


def write_in_block(
        block,
        offset,
        zarr_ds,
        out_ds,
        replace_section_list,
        ):

    out_ds[block.write_roi] = zarr_ds[block.write_roi + offset]

    # logger.info("Running block %s" % block)

    voxel_size = zarr_ds.voxel_size
    zarr_roi = block.write_roi + offset
    zarr_offset = zarr_roi.get_offset()
    roi_shape = zarr_roi.get_shape()

    section_roi_ref = daisy.Roi((0, zarr_offset[1], zarr_offset[2]), (voxel_size[0], roi_shape[1], roi_shape[2]))
    # print(section_roi_ref)
    for s, r in replace_section_list:

        section_roi = section_roi_ref.shift((s*voxel_size[0], 0, 0))
        if zarr_roi.intersects(section_roi):

            replace_section_roi = section_roi_ref.shift((r*voxel_size[0], 0, 0))
            out_ds[section_roi-offset] = zarr_ds[replace_section_roi]
            # logger.info("Fixing section %d by writing %s with %s" % (s, section_roi-offset, replace_section_roi))

    return 0


if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)

    in_config = config["ZarrIn"]
    raw_f = in_config["file"]
    voxel_size = in_config["voxel_size"]

    raw_ds = daisy.open_ds(raw_f, in_config["raw_ds"])

    roi_offset = in_config["roi_offset"]
    if in_config["roi_offset_encoding"] == "voxel":
        roi_offset = [m*n for m, n in zip(roi_offset, voxel_size)]
    elif in_config["roi_offset_encoding"] == "nm":
        pass
    else:
        raise RuntimeError("Currently only support `voxel` or `nm` roi_offset_encoding")

    roi_context = daisy.Coordinate(in_config.get("roi_context_nm", [0, 0, 0]))
    roi_shape = daisy.Coordinate(in_config["roi_shape_nm"])
    roi_offset = daisy.Coordinate(roi_offset)
    print("roi_shape: ", roi_shape)
    print("roi_context: ", roi_context)

    if in_config["center_roi_offset"]:
        roi_offset = roi_offset - roi_shape/2

    roi_shape = roi_shape + roi_context*2
    roi_offset = roi_offset - roi_context

    # out_config = config["zarr"]
    # cutout_f = out_config["dir"] + "/" + script_name + ".zarr"
    cutout_f = config["out_file"]

    write_size = daisy.Coordinate([1000, 1024, 1024])

    total_roi = daisy.Roi((0, 0, 0), roi_shape)

    replace_section_list = in_config.get("replace_section_list", [])

    print("roi_offset: ", roi_offset)
    print("roi_shape: ", roi_shape)
    print("roi_context: ", roi_context)
    print("total_roi: ", total_roi)

    out_ds = daisy.prepare_ds(
        cutout_f,
        'volumes/raw',
        total_roi,
        voxel_size,
        np.uint8,
        write_size=write_size,
        force_exact_write_size=True,
        compressor=None,
        delete=True
    )

    write_roi = daisy.Roi((0, 0, 0), write_size)
    read_roi = write_roi
    print("read_roi: %s" % read_roi)
    print("write_roi: %s" % write_roi)

    # process block-wise
    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: write_in_block(
            b,
            roi_offset,
            raw_ds,
            out_ds,
            replace_section_list,
            ),
        # check_function=lambda b: check_block(
        #     b, segmentation_dss[5]),
        num_workers=12,
        # num_workers=1,
        read_write_conflict=False,
        fit='valid')

    # if "replace_section_list" in in_config and len(in_config["replace_section_list"]):

    #     out_array = out_ds.to_ndarray()

    #     section_roi_ref = daisy.Roi((0, 0, 0), (0, roi_shape[1], roi_shape[2]))
    #     raw_roi = daisy.Roi(roi_offset, roi_shape)
    #     for s, r in in_config["replace_section_list"]:
    #         section_roi = section_roi_ref.shift((s*voxel_size[0], 0, 0))
    #         if raw_roi.intersects(section_roi):
    #             replace_section_roi = section_roi_ref.shift((r*voxel_size[0], 0, 0))

