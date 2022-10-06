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


def mult_coord(coord, voxel_size_xyz):
    return [
        coord[0] * voxel_size_xyz[2],
        coord[1] * voxel_size_xyz[1],
        coord[2] * voxel_size_xyz[0],
    ]


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

    return parse_skeleton_wk(
        fin,
        voxel_size_xyz=config["voxel_size_xyz"],
        coord_in_zyx=config["coord_in_zyx"],
        coord_in_pix=config["coord_in_pix"],
        interpolation=config["interpolation"],
        interpolation_steps=config["interpolation_steps"],
    )


def parse_skeleton_wk(
    fin,
    voxel_size_xyz,
    coord_in_zyx,
    coord_in_pix,
    interpolation=True,
    interpolation_steps=10,
):
    logger.info(f"Parsing {fin}")

    assert coord_in_zyx, "Unimplemented"
    assert coord_in_pix, "Unimplemented"
    skeletons = {}
    nodes = {}

    wk_skels = webknossos.skeleton.Skeleton.load(fin)

    for tree in wk_skels.trees:

        skeleton = []

        def add_node(pos):
            node_ = {"zyx": mult_coord(pos, voxel_size_xyz)}
            node_id = len(nodes)
            nodes[node_id] = node_
            skeleton.append(node_id)

        # add nodes
        for node in tree.nodes():
            add_node(node.position)

        # add interpolated nodes
        if interpolation:
            for edge in tree.edges():
                pos0 = edge[0].position
                pos1 = edge[1].position
                points = interpolate_points(pos0, pos1, max_steps=interpolation_steps)
                for p in points:
                    add_node(p)

        if len(skeleton) == 1:
            # skip single node skeletons (likely to be synapses annotation)
            continue

        skeletons[tree.id] = skeleton

    return skeletons, nodes


def interpolate_points(p0, p1, max_steps):
    delta = []
    for i in range(3):
        delta.append((float(p1[i]) - p0[i]) / max_steps)
    res = []
    for i in range(max_steps - 1):
        res.append(tuple([int(p0[k] + (i + 1) * delta[k]) for k in range(3)]))
    res = list(set(res))
    return res


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


def nm2px(coord, voxel_size, offset, max_shape=None):
    # removes offset and converts to px
    if max_shape is None:
        return [int((c / v) - o) for c, v, o in zip(coord, voxel_size, offset)]
    else:
        return [
            min(s - 2, int((c / v) - o))
            for c, v, o, s in zip(coord, voxel_size, offset, max_shape)
        ]


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
            return image

        except:
            pass

    logger.info(f"Rasterizing skeleton...")

    # Skel=tree_id:[Node_id], nodes=Node_id:{x,y,z}
    skeletons, nodes = parse_skeleton(config_path)

    # {Tree_id:[[xyz]]}
    skel_zyx = {
        tree_id: [nodes[nid]["zyx"] for nid in node_id]
        for tree_id, node_id in skeletons.items()
    }

    # Initialize rasterized skeleton image
    dataset_shape = np.array(config["dataset_shape"])
    voxel_size = config["voxel_size_xyz"]
    offset = config["dataset_offset"]
    image = np.zeros(dataset_shape, dtype=np.uint)

    px = partial(nm2px, voxel_size=voxel_size, offset=offset, max_shape=dataset_shape)
    for id, tree in skel_zyx.items():
        # iterates through ever node and assigns id to {image}
        for i in range(0, len(tree) - 1, 2):
            line = line_nd(px(tree[i]), px(tree[i + 1]))
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
