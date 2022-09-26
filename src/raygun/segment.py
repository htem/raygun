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


def mutex_segment(config_path):
    seg_config = {
        "aff_ds": "pred_affs",
        "prefix": "volumes",
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
    prefix = seg_config["prefix"]
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

    f = zarr.open(file, "a")[prefix]

    # crop a few sections off for context
    affs = f[
        aff_ds
    ]  # [:, crop:-crop, :, :]  # TODO: Ask Arlo about crop # TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0

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


def segment(config_path):
    seg_config = {
        "aff_ds": "pred_affs",
        "thresholds": [t for t in np.arange(0.1, 0.9, 0.1)],
        "prefix": "volumes",
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
        prefix = seg_config["prefix"]
        max_affinity_value = seg_config["max_affinity_value"]
        labels_mask = seg_config["labels_mask"]

        done = True
        for thresh in thresholds:
            done = done and os.path.exists(
                os.path.join(file, prefix, "pred_seg_%.2f" % thresh)
            )

        f = zarr.open(file)[prefix]

        if not done:
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
            # seg_dict = {}
            for thresh, pred_seg in zip(thresholds, pred_segs):
                # seg_dict[thresh] = pred_seg
                f[f'pred_seg_{"{:.2f}".format(thresh)}'] = pred_seg
                f[f'pred_seg_{"{:.2f}".format(thresh)}'].attrs["offset"] = f[
                    aff_ds
                ].attrs["offset"]
                f[f'pred_seg_{"{:.2f}".format(thresh)}'].attrs["resolution"] = f[
                    aff_ds
                ].attrs["resolution"]

        # else:
        #     seg_dict = {}
        #     for thresh in thresholds:
        #         seg_dict[thresh] = f[f'pred_seg_{"{:.2f}".format(thresh)}']

        # return seg_dict


# TODO: MAKE DAISY COMPATIBLE BEFORE 0.3.0
if __name__ == "__main__":
    segment(sys.argv[1])
