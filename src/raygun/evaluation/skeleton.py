from glob import glob
import sys
from raygun.read_config import read_config

import os

from daisy import Coordinate

from raygun.webknossos_utils.wkw_seg_to_zarr import download_wk_skeleton


def mult_coord(coord, voxel_size_xyz):
    return [
        coord[0] * voxel_size_xyz[2],
        coord[1] * voxel_size_xyz[1],
        coord[2] * voxel_size_xyz[0],
    ]


def parse_skeleton(config):
    fin = config["file"]
    if fin.endswith(".zip"):
        return parse_skeleton_wk(
            fin,
            voxel_size_xyz=config["voxel_size_xyz"],
            coord_in_zyx=config["coord_in_zyx"],
            coord_in_pix=config["coord_in_pix"],
            interpolation=config["interpolation"],
            interpolation_steps=config["interpolation_steps"],
        )
    else:
        assert False, "CATMAID NOT IMPLEMENTED"


def parse_skeleton_wk(
    fin,
    voxel_size_xyz,
    coord_in_zyx,
    coord_in_pix,
    interpolation=True,
    interpolation_steps=10,
):

    import webknossos

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


def get_updated_skeleton(default_config_fn=None):
    if default_config_fn is None:
        try:
            default_config_fn = sys.argv[1]
        except:
            default_config_fn = "skeleton.json"
    path = os.path.dirname(os.path.realpath(default_config_fn))
    print(f"Path: {path}")
    segment_config = read_config(default_config_fn)

    if not os.path.exists(segment_config["skeleton_config"]["file"]):
        files = glob(os.path.join(path, "/skeletons/*"))
        if len(files) == 0 or segment_config["skeleton_config"]["file"] == "update":
            skel_file = download_wk_skeleton(
                segment_config["skeleton_config"]["url"].split("/")[-1],
                os.path.join(path, "skeletons/"),
                overwrite=True,
            )
        else:
            skel_file = max(files, key=os.path.getctime)
    skel_file = os.path.abspath(skel_file)

    return skel_file
