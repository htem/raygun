#%%
from functools import partial
from glob import glob
import numpy as np
import sys
import os
import webknossos
from skimage.draw import line_nd
import daisy
from raygun.read_config import read_config
from raygun.webknossos_utils.wkw_seg_to_zarr import download_wk_skeleton

import logging

logger = logging.getLogger(__name__)


def parse_skeleton(config_path):
    logger.info(f"Parsing skeleton...")
    config = read_config(config_path)
    fin = config["file"]
    if not fin.endswith(".zip"):
        try:
            fin = get_updated_skeleton(config_path)
            assert fin.endswith(".zip"), "Skeleton zip file not found."
        except:
            assert False, "CATMAID NOT IMPLEMENTED"

    wk_skels = webknossos.skeleton.Skeleton.load(fin)
    # return wk_skels

    skel_coor = {}
    for tree in wk_skels.trees:
        skel_coor[tree.id] = []
        for start, end in tree.edges.keys():
            start_pos = start.position.to_np()
            end_pos = end.position.to_np()
            skel_coor[tree.id].append([start_pos, end_pos])

    return skel_coor


def get_updated_skeleton(config_path=None):
    if config_path is None:
        try:
            config_path = sys.argv[1]
        except:
            config_path = "skeleton.json"
    config = read_config(config_path)
    if "skeleton_config" in config.keys():
        config = config["skeleton_config"]

    if not os.path.exists(config["file"]):
        if "search_path" in config.keys():
            search_path = config["search_path"].rstrip("/*") + "/*"
        else:
            path = os.path.dirname(os.path.realpath(config_path))
            logger.info(f"Path: {path}")
            search_path = os.path.join(path, "skeletons/*")
        logger.info(f"Search path: {search_path}")
        files = glob(search_path)
        if len(files) == 0 or config["file"] == "update":
            skel_file = download_wk_skeleton(
                config["url"].split("/")[-1],
                search_path.rstrip("*"),
                overwrite=True,
            )
        else:
            skel_file = max(files, key=os.path.getctime)
    skel_file = os.path.abspath(skel_file)

    return skel_file


def rasterize_skeleton(config_path=None):
    if config_path is None:
        config_path = sys.argv[1]

    config = read_config(config_path)

    if "dataset_name" in config.keys() and "." in config["file"]:
        # Load pre-rasterized
        try:
            logger.info("Trying to load skeleton...")
            ds = daisy.open_ds(config["file"], config["dataset_name"])
            image = ds.to_ndarray(ds.roi)
            logger.info("Loaded skeleton...")
            return image

        except:
            logger.warn("Failed to load skeleton...")

    logger.info(f"Rasterizing skeleton...")

    # Skel=tree_id:[Node_id], nodes=Node_id:{x,y,z}
    skel_coor = parse_skeleton(config_path)

    # Initialize rasterized skeleton image
    dataset_shape = np.array(config["dataset_shape"])
    voxel_size = config["voxel_size_xyz"]
    offset = np.array(config["dataset_offset"])
    image = np.zeros(dataset_shape, dtype=np.uint)

    def adjust(coor):
        return np.min([coor - offset, dataset_shape - 1], 0)

    for id, tree in skel_coor.items():
        # iterates through ever node and assigns id to {image}
        for start, end in tree:
            line = line_nd(adjust(start), adjust(end))
            image[line] = id

    if "save_path" in config.keys() and "save_ds" in config.keys():
        logger.info("Saving rasterization...")
        # Save GT rasterization #TODO: implement daisy blockwise option
        total_roi = daisy.Roi(
            daisy.Coordinate(offset) * daisy.Coordinate(voxel_size),
            daisy.Coordinate(dataset_shape) * daisy.Coordinate(voxel_size),
        )
        write_size = daisy.Coordinate((64, 64, 64)) * daisy.Coordinate(
            voxel_size
        )  # TODO: unhardcode
        out_ds = daisy.prepare_ds(
            config["save_path"],
            config["save_ds"],
            total_roi,
            voxel_size,
            image.dtype,
            delete=True,
            write_size=write_size,
        )
        out_ds[out_ds.roi] = image

    return image


# %%
