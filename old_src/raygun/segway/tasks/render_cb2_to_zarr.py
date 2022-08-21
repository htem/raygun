import hashlib
import json
import logging
import numpy as np
import os
import sys
# import time

import daisy

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)

def render_blockwise():
        
    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):
        raw_dataset (``string``):
        lsds_file (``string``):
        lsds_dataset (``string``):

            Paths to the input datasets. lsds can be None if not needed.

        out_file (``string``):
        out_dataset (``string``):

            Path to the output datset.

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''


    # raw_file = os.path.abspath(raw_file)
    # out_file = os.path.abspath(out_file)

    # print('Input file path: ', raw_file)
    # print('Output file path: ', out_file)
    # # from here on, all values are in world units (unless explicitly mentioned)

    # # get ROI of source
    # try:
    #     source = daisy.open_ds(raw_file, raw_dataset)
    # except:
    #     in_dataset = in_dataset + '/s0'
    #     source = daisy.open_ds(in_file, in_dataset)
    # print("Source dataset has shape %s, ROI %s, voxel size %s"%(
    #     source.shape, source.roi, source.voxel_size))
)

    # print("Following ROIs in world units:")
    # print("Total input ROI  = %s"%input_roi)
    # print("Block read  ROI  = %s"%block_read_roi)
    # print("Block write ROI  = %s"%block_write_roi)
    # print("Total output ROI = %s"%output_roi)

    raw_file = '/n/groups/htem/temcagt/datasets/cb2/intersection/cb2_full_volume.zarr'
    raw_ds = daisy.open_ds(raw_file, 'r')
    raw_ds_shape_x = raw_ds.shape[2]

    roi_offset = [375*40, 0, 0]
    roi_shape = [125*40, 73728*4, raw_ds_shape_x]
    total_roi = daisy.Roi(tuple(roi_offset), tuple(roi_shape))

    write_roi_shape = total_roi
    write_roi_shape[0] = 25*40
    write_roi_shape[1] = 8192*4
    write_roi = daisy.Roi((0, 0, 0), tuple(write_roi_shape))

    print("Starting block-wise processing...")

    # process block-wise
    succeeded = daisy.run_blockwise(
        total_roi,
        write_roi,
        write_roi,
        process_function=lambda b: process_block(b),
        check_function=lambda b: check_block(raw_ds, b),
        num_workers=16,
        read_write_conflict=False,
        fit='overhang')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")


def process_block(block):

    y = block.write_roi.get_begin()[1] / 4
    s = block.write_roi.get_begin()[0] / 40

    s_end = block.write_roi.get_end()[0] / 40
    assert s_end - s == 25
    y_end = block.write_roi.get_end()[1] / 4
    assert y_end - y == 8192

    sbatch_cmd = 'sbatch --job-name="zarr_block_{s}_{y}" /n/groups/htem/temcagt/datasets/cb2/intersection/jobs/render_jobs/render_zarr_block.job {s} {y}'.format(s=s, y=y)

    print(sbatch_cmd)


def check_block(ds, block):

    print("Checking if block %s is complete..."%block.write_roi)

    if ds.roi.intersect(block.write_roi).empty():
        print("Block outside of output ROI")
        return True

    center_values = ds[block.write_roi.get_begin()]
    s = np.sum(center_values)
    print("Sum of center values in %s is %f"%(block.write_roi, s))

    return s != 0


if __name__ == "__main__":

    # config_file = sys.argv[1]

    # with open(config_file, 'r') as f:
    #     config = json.load(f)

    # render_blockwise(**config)
    render_blockwise()
