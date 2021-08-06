import argparse
import daisy
import numpy as np
import re
import skimage.measure
import zarr
import pymongo

# monkey-patch os.mkdirs, due to bug in zarr
import os
prev_makedirs = os.makedirs


def makedirs(name, mode=0o777, exist_ok=False):
    # always ok if exists
    return prev_makedirs(name, mode, exist_ok=True)


os.makedirs = makedirs


def downscale_block(in_array, out_array, factor, block, completion_db):

    dims = len(factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    in_shape = daisy.Coordinate(in_data.shape[-dims:])
    assert in_shape.is_multiple_of(factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,)*n_channels + factor

    if in_data.dtype == np.uint64:
        slices = tuple(slice(k//2, None, k) for k in factor)
        out_data = in_data[slices]
    else:
        out_data = skimage.measure.block_reduce(in_data, factor, np.mean)

    try:
        out_array[block.write_roi] = out_data
    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    if completion_db is not None:
        document = {
            'block_id': block.block_id
        }
        completion_db.insert(document)

    return 0


def check_block(
        block,
        vol_ds,
        completion_db
        ):

    write_roi = vol_ds.roi.intersect(block.write_roi)
    if write_roi.empty():
        return True

    if completion_db.count({'block_id': block.block_id}) >= 1:
        # logger.debug("Skipping block with db check")
        return True

    # quarter = (write_roi.get_end() - write_roi.get_begin()) / 4

    # # check values of center and nearby voxels
    # # if np.sum(vol_ds[write_roi.get_begin() + quarter*1]): return True
    # if np.sum(vol_ds[write_roi.get_begin() + quarter*2]): return True
    # # if np.sum(vol_ds[write_roi.get_begin() + quarter*3]): return True

    return False


def downscale(in_array, out_array, factor, write_size, roi, completion_db=None):

    print("Downsampling by factor %s" % (factor,))

    dims = in_array.roi.dims()
    block_roi = daisy.Roi((0,)*dims, write_size)

    print("Processing ROI %s with blocks %s" % (out_array.roi, block_roi))

    precheck = lambda b: True
    postcheck = lambda b: True

    if completion_db is not None:
        precheck = lambda b: check_block(b, out_array, completion_db)

    daisy.run_blockwise(
        roi,
        block_roi,
        block_roi,
        process_function=lambda b: downscale_block(
            in_array,
            out_array,
            factor,
            b,
            completion_db),
        check_function=(precheck, postcheck),
        read_write_conflict=False,
        num_workers=48,
        max_retries=0,
        fit='shrink')


def create_scale_pyramid(
        in_file, in_ds_name, scales, chunk_shape,
        roi=None, subroi=False,
        db_host=None, db_name=None):

    ds = zarr.open(in_file)

    # make sure in_ds_name points to a dataset
    try:
        daisy.open_ds(in_file, in_ds_name)
    except Exception:
        raise RuntimeError("%s does not seem to be a dataset" % in_ds_name)

    db = None
    if db_host is not None:

        if db_name is None:
            db_name = os.path.split(in_file)[1].split('.')[0]

        db_client = pymongo.MongoClient(db_host)

        print(db_name)

        if db_name in db_client.database_names():
            i = input("Reset completion stats for %s before running? Yes/[No]" % db_name)
            if i == "Yes":
                db_client.drop_database(db_name)

        db = db_client[db_name]

    # exit(0)

    initial_scale = 0
    # print(in_ds_name)
    m = re.match("(.+)/s(\d+)", in_ds_name)
    # print(m)
    if m:

        initial_scale = int(m.group(2))
        ds_name = in_ds_name
        in_ds_name = m.group(1)

    elif not in_ds_name.endswith('/s0'):

        ds_name = in_ds_name + '/s0'

        i = input("Moving %s to %s? Yes/[No]" % (in_ds_name, ds_name))
        if i != "Yes":
            exit(0)
        ds.store.rename(in_ds_name, in_ds_name + '__tmp')
        ds.store.rename(in_ds_name + '__tmp', ds_name)

    else:

        ds_name = in_ds_name
        in_ds_name = in_ds_name[:-3]

    print("Scaling %s by a factor of %s" % (in_file, scales))

    prev_array = daisy.open_ds(in_file, ds_name)

    if chunk_shape is not None:
        chunk_shape = daisy.Coordinate(chunk_shape)
    else:
        chunk_shape = daisy.Coordinate(prev_array.data.chunks)
        print("Reusing chunk shape of %s for new datasets" % (chunk_shape,))

    if prev_array.n_channel_dims == 0:
        num_channels = 1
    elif prev_array.n_channel_dims == 1:
        num_channels = prev_array.shape[0]
    else:
        raise RuntimeError(
            "more than one channel not yet implemented, sorry...")

    if roi is None:
        roi = prev_array.roi

    for scale_num, scale in enumerate(scales):

        if scale_num + 1 > initial_scale:

            try:
                scale = daisy.Coordinate(scale)
            except Exception:
                scale = daisy.Coordinate((scale,)*chunk_shape.dims())

            next_voxel_size = prev_array.voxel_size*scale
            next_total_roi = roi.snap_to_grid(
                next_voxel_size,
                mode='grow')
            next_write_size = chunk_shape*next_voxel_size
            if subroi:
                next_total_roi = next_total_roi.snap_to_grid(
                    next_write_size, mode='grow')

            print("Next voxel size: %s" % (next_voxel_size,))
            print("Next total ROI: %s" % next_total_roi)
            print("Next chunk size: %s" % (next_write_size,))

            next_ds_name = in_ds_name + '/s' + str(scale_num + 1)
            print("Preparing %s" % (next_ds_name,))

            try:
                next_array = daisy.open_ds(in_file, next_ds_name, mode='r+')
                assert next_array.roi.contains(next_total_roi)
            except:
                next_array = daisy.prepare_ds(
                    in_file,
                    next_ds_name,
                    total_roi=next_total_roi,
                    voxel_size=next_voxel_size,
                    write_size=next_write_size,
                    dtype=prev_array.dtype,
                    num_channels=num_channels)

            completion_db = None
            if db:
                collection_name = str(scale_num + 1)

                if collection_name not in db.list_collection_names():
                    completion_db = db[collection_name]
                    completion_db.create_index(
                        [('block_id', pymongo.ASCENDING)],
                        name='block_id')
                else:
                    completion_db = db[collection_name]

            downscale(prev_array, next_array, scale, next_write_size, next_total_roi,
                completion_db=completion_db)

            prev_array = next_array
            roi = next_total_roi


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create a scale pyramide for a zarr/N5 container.")

    parser.add_argument(
        '--file',
        '-f',
        type=str,
        help="The input container")
    parser.add_argument(
        '--ds',
        '-d',
        type=str,
        help="The name of the dataset")
    parser.add_argument(
        '--scales',
        '-s',
        nargs='*',
        type=int,
        required=True,
        help="The downscaling factor between scales")
    parser.add_argument(
        '--chunk_shape',
        '-c',
        nargs='*',
        type=int,
        default=None,
        help="The size of a chunk in voxels")

    args = parser.parse_args()

    create_scale_pyramid(args.file, args.ds, args.scales, args.chunk_shape)
