import numpy as np
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


def get_segmentation(affinities, threshold, labels_mask=None):

    fragments = watershed_from_affinities(affinities, labels_mask=labels_mask)[0]

    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    return segmentation


def mutex_segment(path, crop=10, max_affinity_value=1.0, sep=3):

    f = zarr.open(path, "a")

    # crop a few sections off for context
    affs = f["pred_affs"][:, crop:-crop, :, :]  # TODO: Ask Arlo
    offset = f["pred_affs"].attrs["offset"]
    resolution = f["pred_affs"].attrs["resolution"]

    # use average affs to mask
    mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value

    affs = 1 - affs

    affs[:sep] = affs[:sep] * -1
    affs[:sep] = affs[:sep] + 1

    neighborhood = np.array(
        [
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
        ]
    )

    pos_diag = np.round(8 * np.sin(np.linspace(0, np.pi, num=8, endpoint=False)))
    neg_diag = np.round(8 * np.cos(np.linspace(0, np.pi, num=8, endpoint=False)))
    stacked_diag = np.stack([0 * pos_diag, pos_diag, neg_diag], axis=-1)
    neighborhood = np.concatenate([neighborhood, stacked_diag]).astype(np.int8)

    seg = compute_mws_segmentation(
        affs, neighborhood, sep, strides=[1, 10, 10], mask=mask
    )

    f["mutex"] = seg
    f["mutex"].attrs["offset"] = [10 * resolution[0]] + offset[1:]
    f["mutex"].attrs["resolution"] = resolution


if __name__ == "__main__":

    f = zarr.open("test.zarr", "a")

    # crop a few sections off for context
    affs = f["pred_affs"][0:3, 10:90, :, :]
    offset = f["pred_affs"].attrs["offset"]
    resolution = f["pred_affs"].attrs["resolution"]

    seg = get_segmentation(affs[:], 0.2)

    f["seg"] = seg
    f["seg"].attrs["offset"] = [10 * resolution[0]] + offset[1:]
    f["seg"].attrs["resolution"] = resolution
