#%%
from glob import glob
import sys
import os
from daisy import Coordinate
from raygun.read_config import read_config
from raygun.webknossos_utils.wkw_seg_to_zarr import download_wk_skeleton


def mult_coord(coord, voxel_size_xyz):
    return [
        coord[0] * voxel_size_xyz[2],
        coord[1] * voxel_size_xyz[1],
        coord[2] * voxel_size_xyz[0],
    ]


def parse_skeleton(config_path):
    config = read_config(config_path)["skeleton_config"]
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


def get_updated_skeleton(config_path=None):
    if config_path is None:
        try:
            config_path = sys.argv[1]
        except:
            config_path = "skeleton.json"
    path = os.path.dirname(os.path.realpath(config_path))
    print(f"Path: {path}")
    config = read_config(config_path)

    if not os.path.exists(config["skeleton_config"]["file"]):
        if "search_path" in config["skeleton_config"].keys():
            search_path = (
                config["skeleton_config"]["search_path"].rstrip("/*") + "/*"
            )
        else:
            search_path = os.path.join(path, "/skeletons/*")
        print(f"Search path: {search_path}")
        files = glob(search_path)
        if len(files) == 0 or config["skeleton_config"]["file"] == "update":
            skel_file = download_wk_skeleton(
                config["skeleton_config"]["url"].split("/")[-1],
                search_path.rstrip("*"),
                overwrite=True,
            )
        else:
            skel_file = max(files, key=os.path.getctime)
    skel_file = os.path.abspath(skel_file)

    return skel_file


# %%
