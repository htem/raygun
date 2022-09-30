from glob import glob
from logging import getLogger
import os
import sys
import numpy as np
from raygun import read_config
import waterz
import zarr
from scipy.ndimage import label, maximum_filter, measurements, distance_transform_edt
from skimage.segmentation import watershed
from affogato.segmentation import compute_mws_segmentation

import logging

logger = logging.getLogger(__name__)


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

    logger.info(f"Found {n} fragments")

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


def mutex_segment(config_path):
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
        "mask_thresh": 0.5,
    }

    temp = read_config(config_path)
    seg_config.update(temp)

    file = seg_config["file"]
    aff_ds = seg_config["aff_ds"]
    max_affinity_value = seg_config["max_affinity_value"]
    sep = seg_config["sep"]
    n_diagonals = seg_config["n_diagonals"]
    neighborhood = np.array(seg_config["neighborhood"])
    mask_thresh = seg_config["mask_thresh"]

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

    logger.info("Loading affinity predictions...")
    affs = f[aff_ds][:]  # TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0

    # use average affs to mask
    mask = np.mean(affs, axis=0) > mask_thresh * max_affinity_value

    affs = 1 - affs

    affs[:sep] = affs[:sep] * -1
    affs[:sep] = affs[:sep] + 1

    logger.info("Getting segmentations...")
    seg = compute_mws_segmentation(
        affs, neighborhood, sep, strides=[10, 10, 10], mask=mask
    )

    if "save" in seg_config.keys() and not seg_config["save"]:
        return seg

    logger.info("Writing segmentations...")
    if "dest_dataset" not in seg_config.keys():
        dest_dataset = f"mutex_{'{:.2f}'.format(mask_thresh)}"
    else:
        dest_dataset = seg_config["dest_dataset"]
    f[dest_dataset] = seg
    f[dest_dataset].attrs["offset"] = f[aff_ds].attrs["offset"]
    f[dest_dataset].attrs["resolution"] = f[aff_ds].attrs["resolution"]

    view_script = os.path.join(
        os.path.dirname(config_path),
        f"view_{os.path.basename(file).rstrip('.n5').rstrip('.zarr')}.ng",
    )

    if not os.path.exists(view_script):
        with open(view_script, "w") as f:
            f.write(f"neuroglancer -f {file} -d {dest_dataset} ")

    else:
        with open(view_script, "a") as f:
            f.write(f"{dest_dataset} ")


def segment(config_path=None):  # TODO: Clean up
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
        mutex_segment(config_path)

    else:
        file = seg_config["file"]
        thresholds = seg_config["thresholds"]
        aff_ds = seg_config["aff_ds"]
        max_affinity_value = seg_config["max_affinity_value"]
        labels_mask = seg_config["labels_mask"]

        done = True
        for thresh in thresholds:
            done = done and os.path.exists(os.path.join(file, "pred_seg_%.2f" % thresh))

        f = zarr.open(file)

        if not done:
            # load predicted affinities
            logger.info("Loading affinity predictions...")
            prediction = f[aff_ds][:].astype(
                np.float32
            )  # TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0
            logger.info("Getting segmentations...")
            pred_segs = get_segmentation(
                prediction,
                thresholds=thresholds,
                labels_mask=labels_mask,
                max_affinity_value=max_affinity_value,
            )

            # save segmentation
            view_script = os.path.join(
                os.path.dirname(config_path),
                f"view_{os.path.basename(file).rstrip('.n5').rstrip('.zarr')}.ng",
            )
            logger.info("Writing segmentations...")
            for thresh, pred_seg in zip(thresholds, pred_segs):
                dest_dataset = f'pred_seg_{"{:.2f}".format(thresh)}'
                f[dest_dataset] = pred_seg
                f[dest_dataset].attrs["offset"] = f[aff_ds].attrs["offset"]
                f[dest_dataset].attrs["resolution"] = f[aff_ds].attrs["resolution"]

                if not os.path.exists(view_script):
                    with open(view_script, "w") as v:
                        v.write(f"neuroglancer -f {file} -d {dest_dataset} ")

                else:
                    with open(view_script, "a") as v:
                        v.write(f"{dest_dataset} ")


# TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0
if __name__ == "__main__":
    segment(sys.argv[1])
