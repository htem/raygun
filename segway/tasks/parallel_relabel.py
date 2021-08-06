import daisy
import lsd
import logging
import numpy as np

logger = logging.getLogger(__name__)


def relabel_in_block(block, segmentation_ds, fragments_ds, fragments_map):

    logger.info("Relabeling in %s", block.read_roi)
    logger.info("Reading fragment...")
    volume = fragments_ds[block.read_roi].to_ndarray()
    logger.info("Getting and sorting unique fragments...")
    fragments = np.unique(volume)
    logger.info("Making segments...")
    segments = np.array([
        fragments_map.get(fragment, fragment)
        for fragment in fragments
    ], dtype=fragments.dtype)

    if len(fragments) == 0:
        logger.warn("Block %s contains no relevant fragments", block.block_id)
        return

    # shift fragment values to potentially save memory when relabeling
    min_fragment = fragments.min()
    offset = 0
    if min_fragment > 0:
        offset = fragments.dtype.type(min_fragment - 1)
        volume -= offset
        fragments -= offset

    logger.debug("Mapping fragments to %d segments", len(segments))
    volume = lsd.labels.replace_values(volume, fragments, segments)

    logger.debug("Writing relabeled block to segmentation volume")
    segmentation_ds[block.read_roi] = volume

def parallel_relabel(
        fragments_map,
        fragments_file,
        fragments_dataset,
        total_roi,
        block_size,
        seg_file,
        seg_dataset,
        num_workers,
        retry):

    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)

    logger.info("Preparing segmentation dataset...")
    fragments_ds = daisy.open_ds(fragments_file, fragments_dataset)
    segmentation_ds = daisy.prepare_ds(
        seg_file,
        seg_dataset,
        total_roi,
        voxel_size=fragments_ds.voxel_size,
        dtype=np.uint64,
        write_roi=write_roi)

    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: relabel_in_block(
                b,
                segmentation_ds,
                fragments_ds,
                fragments_map),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logger.error("parallel relabel failed, retrying %d/%d", i + 1, retry)