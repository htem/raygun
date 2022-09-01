import daisy
import sys
import logging
import numpy as np
import gt_tools

# from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)

    file = config["file"]

    # mask_ds = daisy.open_ds(
    #     file,
    #     config["mask_ds"])

    # segment_ds_name = config["segment_ds"]
    # if "myelin_mask_file" in config:
    #     segment_ds_name = "volumes/labels/neuron_ids_myelin"
    # elif "skeleton_file" in config:
    #     #segment_ds_name = "volumes/segmentation_skeleton"
    #     segment_ds_name = config["segmentation_skeleton_ds"]

    # segment_ds = daisy.open_ds(
    #     file,
    #     segment_ds_name)

    # if "skeleton_file" in config:
    #     unlabeled_ds = "volumes/labels/unlabeled_mask_skeleton"
    # else:
    #     unlabeled_ds = "volumes/labels/unlabeled_mask"
    # unlabeled_ds = daisy.open_ds(file, unlabeled_ds)

    raw_file = config["raw_file"]
    raw_ds = daisy.open_ds(raw_file, "volumes/raw")

    out_file = config["out_file"]

    # if 'clear_myelin' in config and config['clear_myelin']:
    #     myelin_ds = daisy.open_ds(file, config["myelin_ds"], 'r+')
        # myelin_ds[myelin_ds.roi] = 255

    # myelin_ds = daisy.open_ds(file, config["myelin_ds"])

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/neuron_ids",
            raw_ds.roi,
            raw_ds.voxel_size,
            np.uint64,
            compressor={'id': 'zlib', 'level': 5}
            )
        print("Writing neuron_ids...")
        out[out.roi] = 1

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/labels_mask2",
            raw_ds.roi,
            raw_ds.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5}
            )
        out[out.roi] = 1

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/unlabeled",
            raw_ds.roi,
            raw_ds.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5}
            )
        out[out.roi] = 1

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/roi",
            raw_ds.roi,
            raw_ds.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5}
            )
        out[out.roi] = 1

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/myelin_gt",
            raw_ds.roi,
            raw_ds.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5}
            )
        out[out.roi] = 255

    if True:
        if raw_file != out_file:
            raw_out = daisy.prepare_ds(
                out_file,
                "volumes/raw",
                raw_ds.roi,
                raw_ds.voxel_size,
                raw_ds.dtype,
                compressor=None
                )
            print("Copying raw...")
            raw_out[raw_ds.roi] = raw_ds
