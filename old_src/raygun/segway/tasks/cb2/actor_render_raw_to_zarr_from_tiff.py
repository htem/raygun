import json
import os
import logging
import numpy as np
import sys
import daisy
import pymongo

import cv2

logging.basicConfig(level=logging.INFO)


# adapted from block_switcher.py
# 
def process_block(
        block,
        voxel_size,
        aligned_dir_path,
        zarr_out,
        y_folder_block_size,
        y_tile_block_size,
        x_tile_block_size,
        bad_sections=set(),
        avail_sections=set()
        ):

    print("Processing ", block)

    roi = block.write_roi
    voxel_size = daisy.Coordinate(voxel_size)

    z0, y0, x0 = roi.get_begin() / voxel_size
    # assert y0 % y_folder_block_size == 0
    y_folder = int(y0 / y_folder_block_size) * y_folder_block_size

    col = int(x0 / x_tile_block_size) + 1
    row = int((y0 % y_folder_block_size) / y_tile_block_size) + 1
    abs_row = int(y0 / y_tile_block_size) + 1

    sections = range((roi.get_begin()/voxel_size)[0], (roi.get_end()/voxel_size)[0])
    # print(avail_sections); exit(0)

    arr = np.zeros(roi.get_shape() / voxel_size, dtype=np.uint8)
    # go through and fill block
    for num in sections:
        dir_name = "%s/%.4d" % (aligned_dir_path, num)
        # print(dir_name)
        tif = "%s/%s/c%.2dr%.2d.tif" % (dir_name, y_folder, col, row)
        # print(tif)
        if not os.path.exists(tif):

            # hack because some folders are not rendered in the parallel format
            tif = "%s/c%.2dr%.2d.tif" % (dir_name, col, abs_row)
            if not os.path.exists(tif):
                if num not in bad_sections:
                    raise RuntimeError("Cannot read %s" % tif)
            #     continue
            # if num in avail_sections:
            #     raise RuntimeError("Cannot read %s" % tif)
            # # assert num in bad_sections, "Cannot read %s" % tif
            continue
        i = cv2.imread(tif, 0)
        if i.shape != (y_tile_block_size, x_tile_block_size):
            raise RuntimeError("%s is malformed" % tif)
            # print("%s is malformed" % tif)
            # continue
        arr[num-sections[0], :, :] = i

    print("Writing block %s to ZARR.." % block)
    zarr_out[roi] = arr


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    zarr_out = daisy.open_ds(
        zarr_file, zarr_dataset, 'r+')

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        process_block(
            block,
            voxel_size,
            aligned_dir_path,
            zarr_out,
            y_folder_block_size,
            y_tile_block_size,
            x_tile_block_size,
            set(bad_sections),
            set(avail_sections)
            )

        # recording block done in the database
        document = dict()
        document.update({
            'block_id': block.block_id,
            'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
            'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
            'start': 0,
            'duration': 0
        })
        completion_db.insert(document)

        client_scheduler.release_block(block, ret=0)