import daisy
import sys
import logging
import numpy as np
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

    config = gt_tools.load_config(sys.argv[1])

    file = config["file"]

    mask_ds = daisy.open_ds(
        file,
        config["mask_ds"])

    segment_ds_name = config["segment_ds_paintera_out"]
    # segment_ds_name = config["segment_ds"]
    # if "myelin_mask_file" in config:
    #     segment_ds_name = "volumes/labels/neuron_ids_myelin"
    # elif "skeleton_file" in config:
    #     #segment_ds_name = "volumes/segmentation_skeleton"
    #     segment_ds_name = config["segmentation_skeleton_ds"]

    segment_ds = daisy.open_ds(
        file,
        segment_ds_name)

    if "skeleton_file" in config:
        unlabeled_ds = "volumes/labels/unlabeled_mask_skeleton"
    else:
        unlabeled_ds = "volumes/labels/unlabeled_mask"
    unlabeled_ds = daisy.open_ds(file, unlabeled_ds)

    raw_file = config["raw_file"]
    raw_ds = daisy.open_ds(raw_file, "volumes/raw")

    myelin_ds = daisy.open_ds(file, config["myelin_ds"])
    out_file = config["out_file"]

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/neuron_ids",
            segment_ds.roi,
            segment_ds.voxel_size,
            segment_ds.dtype,
            compressor={'id': 'zlib', 'level': 5}
            )
        print("Reading neuron_ids...")
        out_array = segment_ds.to_ndarray()
        np.place(out_array, myelin_ds.to_ndarray() == 0, 0)
        print("Writing neuron_ids...")
        out[out.roi] = out_array

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/labels_mask",
            mask_ds.roi,
            segment_ds.voxel_size,
            mask_ds.dtype,
            compressor={'id': 'zlib', 'level': 5}
            )
        print("Reading labels_mask...")
        out_array = daisy.Array(
            np.zeros(out.shape, dtype=out.dtype), out.roi, out.voxel_size)

        out_array[mask_ds.roi] = mask_ds
        out_nd = out_array[mask_ds.roi].to_ndarray()
        unlabeled_nd = unlabeled_ds[mask_ds.roi].to_ndarray()
        out_nd = out_nd & unlabeled_nd
        out_array[mask_ds.roi] = out_nd
        print("Writing labels_mask...")
        out[out.roi] = out_array

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/labels_mask2",
            mask_ds.roi,
            segment_ds.voxel_size,
            mask_ds.dtype,
            compressor={'id': 'zlib', 'level': 5}
            )
        print("Copying labels_mask2...")
        out[out.roi] = mask_ds

    if True:
        out = daisy.prepare_ds(
            out_file,
            "volumes/labels/unlabeled",
            unlabeled_ds.roi,
            segment_ds.voxel_size,
            unlabeled_ds.dtype,
            compressor={'id': 'zlib', 'level': 5}
            )
        print("Copying unlabeled...")
        out[out.roi] = unlabeled_ds

    if True:
        if raw_file != out_file:
            raw_out = daisy.prepare_ds(
                out_file,
                "volumes/raw",
                raw_ds.roi,
                segment_ds.voxel_size,
                raw_ds.dtype,
                compressor=None
                )
            print("Copying raw...")
            raw_out[raw_ds.roi] = raw_ds
