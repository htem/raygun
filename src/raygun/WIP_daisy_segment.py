import logging
import os
import sys
import numpy as np
from raygun import read_config
import waterz
import zarr
from scipy.ndimage import label, maximum_filter, measurements, distance_transform_edt
from skimage.segmentation import watershed
from affogato.segmentation import compute_mws_segmentation


def watershed_from_boundary_distance(
    boundary_distances,
    boundary_mask,
    return_seeds=False,
    id_offset=0,
    min_seed_distance=10,
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=boundary_mask
    )

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=10,
    labels_mask=None,
):

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0

        for z in range(depth):

            boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            if labels_mask is not None:

                boundary_mask *= labels_mask.astype(bool)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            return_seeds=return_seeds,
            min_seed_distance=min_seed_distance,
        )

        fragments = ret[0]

    return ret


# %%
# function to extract segmentation
# 1) get supervoxels (fragments) from affinities
# 2) agglomerate
# 3) get next item from generator (seg)
def get_segmentation(affinities, thresholds, labels_mask=None, max_affinity_value=None):

    if max_affinity_value is None:
        max_affinity_value = np.max(affinities)

    fragments = watershed_from_affinities(
        affinities, max_affinity_value=max_affinity_value, labels_mask=labels_mask
    )[0]

    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    generator = waterz.agglomerate(
        affs=affinities,
        fragments=fragments,
        thresholds=thresholds,
        scoring_function="OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
    )

    segmentations = [seg.copy() for seg in generator]

    return segmentations


def mutex_worker(config_path):
    seg_config = {
        "aff_ds": "pred_affs",
        "max_affinity_value": 1.0,
        "sep": 3,
        "neighborhood": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
            [8, 0, 0],
            [0, 8, 0],
            [0, 0, 8],
        ],
        "n_diagonals": 8,
    }

    temp = read_config(config_path)
    seg_config.update(temp)

    file = seg_config["file"]
    aff_ds = seg_config["aff_ds"]
    max_affinity_value = seg_config["max_affinity_value"]
    sep = seg_config["sep"]
    n_diagonals = seg_config["n_diagonals"]
    neighborhood = np.array(seg_config["neighborhood"])

    if n_diagonals > 0:
        pos_diag = np.round(
            n_diagonals * np.sin(np.linspace(0, np.pi, num=n_diagonals, endpoint=False))
        )
        neg_diag = np.round(
            n_diagonals * np.cos(np.linspace(0, np.pi, num=n_diagonals, endpoint=False))
        )
        stacked_diag = np.stack([0 * pos_diag, pos_diag, neg_diag], axis=-1)
        neighborhood = np.concatenate([neighborhood, stacked_diag]).astype(np.int8)

    f = zarr.open(file, "a")

    # crop a few sections off for context
    affs = f[aff_ds][:]  # TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0

    # use average affs to mask
    mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value

    affs = 1 - affs

    affs[:sep] = affs[:sep] * -1
    affs[:sep] = affs[:sep] + 1

    seg = compute_mws_segmentation(
        affs, neighborhood, sep, strides=[10, 10, 10], mask=mask
    )

    f["mutex"] = seg
    f["mutex"].attrs["offset"] = f[aff_ds].attrs["offset"]
    f["mutex"].attrs["resolution"] = f[aff_ds].attrs["resolution"]


def segment_worker(config_path=None):
    client = daisy.Client()
    worker_id = client.worker_id
    logger = logging.getLogger(f"crop_worker_{worker_id}")
    logger.info(f"Launching {worker_id}...")
    if config_path is None:
        config_path = sys.argv[1]

    seg_config = {
        "aff_ds": "pred_affs",
        "thresholds": [t for t in np.arange(0.1, 0.9, 0.1)],
        "mutex": False,
        "max_affinity_value": 1.0,
        "labels_mask": None,
    }

    temp = read_config(config_path)
    seg_config.update(temp)
    if seg_config["mutex"]:
        mutex_worker(config_path)

    else:
        file = seg_config["file"]
        thresholds = seg_config["thresholds"]
        aff_ds = seg_config["aff_ds"]
        max_affinity_value = seg_config["max_affinity_value"]
        labels_mask = seg_config["labels_mask"]

        f = zarr.open(file)

        # load predicted affinities
        prediction = f[aff_ds][:].astype(
            np.float32
        )  # TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0
        pred_segs = get_segmentation(
            prediction,
            thresholds=thresholds,
            labels_mask=labels_mask,
            max_affinity_value=max_affinity_value,
        )

        # save segmentation
        for thresh, pred_seg in zip(thresholds, pred_segs):
            f[f'pred_seg_{"{:.2f}".format(thresh)}'] = pred_seg
            f[f'pred_seg_{"{:.2f}".format(thresh)}'].attrs["offset"] = f[aff_ds].attrs[
                "offset"
            ]
            f[f'pred_seg_{"{:.2f}".format(thresh)}'].attrs["resolution"] = f[
                aff_ds
            ].attrs["resolution"]

    while True:
        with client.acquire_block() as block:
            if block is None:
                break

            else:
                prediction = source.to_ndarray(block.read_roi)
                destination[block.write_roi] = out


#%%
def segment(segment_config_path=None):  # Use absolute path
    if segment_config_path is None:
        segment_config_path = sys.argv[1]

    logger = logging.getLogger(__name__)
    logger.info("Loading segmentation config...")

    segment_config = {  # Defaults
        "crop": 0,
        "read_size": None,
        "max_retries": 2,
        "num_workers": 16,
        "ndims": None,
        "net_name": None,
        "output_ds": None,
        "out_specs": None,
    }

    temp = read_config(segment_config_path)
    segment_config.update(temp)

    config_path = segment_config["config_path"]
    train_config = read_config(config_path)
    source_path = segment_config["source_path"]
    source_dataset = segment_config["source_dataset"]
    net_name = segment_config["net_name"]
    checkpoint = segment_config["checkpoint"]
    # compressor = segment_config['compressor']
    num_workers = segment_config["num_workers"]
    max_retries = segment_config["max_retries"]
    output_ds = segment_config["output_ds"]
    out_specs = segment_config["out_specs"]
    ndims = segment_config["ndims"]
    if ndims is None:
        ndims = train_config["ndims"]

    dest_path = os.path.join(
        os.path.dirname(config_path), os.path.basename(source_path)
    )
    if output_ds is None:
        if net_name is not None:
            output_ds = [f"{source_dataset}_{net_name}_{checkpoint}"]
        else:
            output_ds = [f"{source_dataset}_{checkpoint}"]

    source = daisy.open_ds(source_path, source_dataset)

    # Get input/output sizes #TODO: Clean this up with refactor prior to 0.3.0 (will break old CGAN configs...)
    if "input_shape" in segment_config.keys() or "input_shape" in train_config.keys():
        try:
            input_shape = segment_config["input_shape"]
            output_shape = segment_config["output_shape"]
        except:
            input_shape = train_config["input_shape"]
            output_shape = train_config["output_shape"]

        if not isinstance(input_shape, list):
            input_shape = daisy.Coordinate(
                (1,) * (3 - ndims) + (input_shape,) * (ndims)
            )
            output_shape = daisy.Coordinate(
                (1,) * (3 - ndims) + (output_shape,) * (ndims)
            )
        else:
            input_shape = daisy.Coordinate(input_shape)
            output_shape = daisy.Coordinate(output_shape)

        read_size = input_shape * source.voxel_size
        write_size = output_shape * source.voxel_size
        context = (read_size - write_size) // 2
        read_roi = daisy.Roi((0, 0, 0), read_size)
        write_roi = daisy.Roi(context, write_size)

    else:
        read_size = segment_config["read_size"]  # CHANGE TO input_shape
        if read_size is None:
            read_size = train_config["side_length"]  # CHANGE TO input_shape
        crop = segment_config["crop"]
        read_size = daisy.Coordinate((1,) * (3 - ndims) + (read_size,) * (ndims))
        crop = daisy.Coordinate((0,) * (3 - ndims) + (crop,) * (ndims))

        read_roi = daisy.Roi([0, 0, 0], source.voxel_size * read_size)
        write_size = read_size - crop * 2
        write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)

    # Prepare output datasets
    for dest_dataset in output_ds:
        these_specs = {
            "filename": dest_path,
            "ds_name": dest_dataset,
            "total_roi": source.data_roi,
            "voxel_size": source.voxel_size,
            "dtype": source.dtype,
            "write_size": write_roi.get_shape(),
            "num_channels": 1,
            "delete": True,
        }
        if out_specs is not None and dest_dataset in out_specs.keys():
            these_specs.update(out_specs[dest_dataset])

        destination = daisy.prepare_ds(**these_specs)

    # Make temporary directory for storing log files
    with tempfile.TemporaryDirectory() as temp_dir:
        cur_dir = os.getcwd()
        os.chdir(temp_dir)
        print(f"Executing in {os.getcwd()}")

        if "launch_command" in segment_config.keys():
            launch_command = [
                *segment_config["launch_command"].split(" "),
                "python",
                import_module(
                    ".".join(["raygun", train_config["framework"], "predict", "worker"])
                ).__file__,
                segment_config_path,
            ]
            process_function = lambda: Popen(launch_command)
            logger.info(f"Launch command: {' '.join(launch_command)}")

        else:
            worker = getattr(
                import_module(
                    ".".join(["raygun", train_config["framework"], "predict", "worker"])
                ),
                "worker",
            )
            process_function = lambda: worker(segment_config_path)

        task = daisy.Task(
            os.path.basename(segment_config_path).rstrip(".json"),
            total_roi=source.data_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            read_write_conflict=True,
            # fit="shrink",
            num_workers=num_workers,
            max_retries=max_retries,
            process_function=process_function,
        )

        logger.info("Running blockwise prediction...")
        if daisy.run_blockwise([task]):
            logger.info("Daisy done.")
        else:
            raise ValueError("Daisy failed.")

        logger.info("Saving viewer script...")
        view_script = os.path.join(
            os.path.dirname(config_path),
            f"view_{os.path.basename(source_path).rstrip('.n5').rstrip('.zarr')}.ng",
        )

        # Add each datasets to viewing file
        for dest_dataset in output_ds:
            if not os.path.exists(view_script):
                with open(view_script, "w") as f:
                    f.write(
                        f"neuroglancer -f {source_path} -d {source_dataset} -f {dest_path} -d {dest_dataset} "
                    )

            else:
                with open(view_script, "a") as f:
                    f.write(f"{dest_dataset} ")

        logger.info("Done.")

    os.chdir(cur_dir)


# TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0
if __name__ == "__main__":
    segment(sys.argv[1])
