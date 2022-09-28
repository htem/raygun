import collections
from raygun.utils import load_json_file, merge_dicts

import copy
import datetime
import os

from daisy import Coordinate


def aggregate_configs(configs):

    input_config = configs["Input"]
    global_config = configs.get("GlobalConfig", {})
    network_config = configs["Network"]

    if "xy_downsample" in input_config:
        print("`Input.xy_downsample` is deprecated!")
        print("Please use `Input.zyx_downsample`")
    if "xy_downsample" in network_config:
        print("`Network.xy_downsample` is deprecated!")
        print("Please use `Input.zyx_downsample`")

    today = datetime.date.today()
    parameters = {}
    parameters["year"] = today.year
    parameters["month"] = "%02d" % today.month
    parameters["day"] = "%02d" % today.day
    parameters["network"] = network_config["name"]
    parameters["iteration"] = network_config["iteration"]
    config_filename = input_config["config_filename"]

    parameters["proj"] = input_config.get("proj", "")
    if parameters["proj"] == "":
        # proj is just the last folder in the config path
        parameters["proj"] = config_filename.split("/")[-2]

    script_name = config_filename.split("/")[-1].split(".")
    if len(script_name) > 2:
        raise RuntimeError("script_name name %s cannot have more than two `.`")
    else:
        script_name = script_name[0]
    parameters["script_name"] = script_name
    parameters["script_folder"] = parameters["proj"]
    parameters["script_dir"] = "/".join(config_filename.split("/")[0:-1])
    script_dir = parameters["script_dir"]

    input_config["experiment"] = input_config["experiment"].format(**parameters)
    parameters["experiment"] = input_config["experiment"]

    # input_config["output_file"] = input_config["output_file"].format(**parameters)

    input_config_synful = copy.deepcopy(input_config)
    input_config_synful1 = copy.deepcopy(input_config)

    for config in input_config:
        if isinstance(input_config[config], str):
            input_config[config] = input_config[config].format(**parameters)

    configs["output_file"] = input_config["output_file"]
    configs["synful_output_file"] = input_config_synful["output_file"]
    configs["synful_output_file1"] = input_config_synful1["output_file"]

    for path_name in ["output_file", "synful_output_file", "synful_output_file1"]:

        output_path = configs[path_name]
        if not os.path.exists(output_path):
            output_path = os.path.join(script_dir, output_path)
        output_path = os.path.abspath(output_path)
        if output_path.startswith("/mnt/orchestra_nfs/"):
            output_path = output_path[len("/mnt/orchestra_nfs/") :]
            output_path = "/n/groups/htem/" + output_path

    os.makedirs(input_config["log_dir"], exist_ok=True)

    merge_function = configs["AgglomerateTask"]["merge_function"]
    thresholds_lut = configs["GlobalConfig"]["thresholds"]

    voxel_size = Coordinate(configs["Input"]["voxel_size"])

    def mult_voxel(config, key):
        if key in config:
            config[key] = Coordinate(config[key])
            config[key] *= voxel_size

    if "Input" in configs:
        config = configs["Input"]
        if config.get("size_in_pix", False):
            mult_voxel(config, "sub_roi_offset")
            mult_voxel(config, "sub_roi_shape")
            mult_voxel(config, "roi_offset")
            mult_voxel(config, "roi_shape")
            mult_voxel(config, "roi_context")

    for config in configs:

        if "Task" not in config:
            # print("Skipping %s" % config)
            continue

        config = configs[config]
        copy_parameter(input_config, config, "db_name")
        copy_parameter(input_config, config, "db_host")
        copy_parameter(input_config, config, "log_dir")
        copy_parameter(input_config, config, "sub_roi_offset")
        copy_parameter(input_config, config, "sub_roi_shape")

        if "num_workers" in config:
            config["num_workers"] = int(config["num_workers"])

    if "GlobalConfig" in configs:
        config = configs["GlobalConfig"]
        if config.get("block_size_in_pix", False):
            mult_voxel(config, "fragments_block_size")
            mult_voxel(config, "fragments_context")
            mult_voxel(config, "agglomerate_block_size")
            mult_voxel(config, "agglomerate_context")
            mult_voxel(config, "find_segments_block_size")
            mult_voxel(config, "write_size")

    if "PredictTask" in configs:
        config = configs["PredictTask"]
        config["raw_file"] = input_config["raw_file"]
        config["raw_dataset"] = input_config["raw_dataset"]
        if "out_file" not in config:
            config["out_file"] = input_config["output_file"]
        config["train_dir"] = network_config["train_dir"]
        config["iteration"] = network_config["iteration"]
        copy_parameter(network_config, config, "net_voxel_size")
        copy_parameter(network_config, config, "predict_file")
        if "predict_file" not in config or config["predict_file"] is None:
            config["predict_file"] = "predict.py"
        copy_parameter(input_config, config, "zyx_downsample")
        if "roi_offset" in input_config:
            config["roi_offset"] = input_config["roi_offset"]
        if "roi_shape" in input_config:
            config["roi_shape"] = input_config["roi_shape"]
        if "roi_context" in input_config:
            config["roi_context"] = input_config["roi_context"]
        copy_parameter(input_config, config, "delete_section_list")
        copy_parameter(input_config, config, "replace_section_list")
        copy_parameter(input_config, config, "overwrite_sections")
        copy_parameter(input_config, config, "overwrite_mask_f")
        copy_parameter(input_config, config, "center_roi_offset")
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(input_config, config, "roi_shrink_context")
        copy_parameter(input_config, config, "roi_context")
        copy_parameter(network_config, config, "output_key")

    if "FixRawFromCatmaidTask" in configs:
        config = configs["FixRawFromCatmaidTask"]
        copy_parameter(input_config, config, "raw_file")
        copy_parameter(input_config, config, "raw_dataset")

    if "DownsampleTask" in configs:
        config = configs["DownsampleTask"]
        copy_parameter(input_config, config, "output_file", "affs_file")

    if "ExtractFragmentTask" in configs:
        config = configs["ExtractFragmentTask"]
        copy_parameter(input_config, config, "output_file", "affs_file")
        copy_parameter(input_config, config, "output_file", "fragments_file")
        copy_parameter(input_config, config, "raw_file")
        copy_parameter(input_config, config, "raw_dataset")
        copy_parameter(input_config, config, "overwrite_sections")
        copy_parameter(input_config, config, "overwrite_mask_f")
        copy_parameter(input_config, config, "db_file_name")
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(global_config, config, "fragments_block_size", "block_size")
        copy_parameter(global_config, config, "fragments_context", "context")

    if "AgglomerateTask" in configs:
        config = configs["AgglomerateTask"]
        if "affs_file" not in config:
            config["affs_file"] = input_config["output_file"]
        config["fragments_file"] = input_config["output_file"]
        config["merge_function"] = merge_function
        copy_parameter(input_config, config, "sub_roi_offset")
        copy_parameter(input_config, config, "sub_roi_shape")
        copy_parameter(input_config, config, "overwrite_sections")
        copy_parameter(input_config, config, "overwrite_mask_f")
        config["edges_collection"] = "edges_" + merge_function
        copy_parameter(input_config, config, "db_file_name")
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(
            global_config, config, "fragments_block_size", "filedb_nodes_chunk_size"
        )
        copy_parameter(global_config, config, "agglomerate_block_size", "block_size")
        copy_parameter(
            global_config, config, "agglomerate_block_size", "filedb_edges_chunk_size"
        )
        copy_parameter(global_config, config, "agglomerate_context", "context")

    if "FindSegmentsGetLocalLUTsTask" in configs:
        config = configs["FindSegmentsGetLocalLUTsTask"]
        copy_parameter(input_config, config, "output_file", "fragments_file")
        config["merge_function"] = merge_function
        config["edges_collection"] = "edges_" + merge_function
        if "thresholds" not in config:
            config["thresholds"] = thresholds_lut
        copy_parameter(input_config, config, "db_file_name")
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(
            global_config, config, "fragments_block_size", "filedb_nodes_chunk_size"
        )
        copy_parameter(
            global_config, config, "agglomerate_block_size", "filedb_edges_chunk_size"
        )
        copy_parameter(global_config, config, "find_segments_block_size", "block_size")

    if "FindSegmentsGetLocalEdgesTask" in configs:
        config = configs["FindSegmentsGetLocalEdgesTask"]
        copy_parameter(input_config, config, "output_file", "fragments_file")
        config["merge_function"] = merge_function
        if "thresholds" not in config:
            config["thresholds"] = thresholds_lut
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(global_config, config, "find_segments_block_size", "block_size")

    if "FindSegmentsGetGlobalLUTsTask" in configs:
        config = configs["FindSegmentsGetGlobalLUTsTask"]
        copy_parameter(input_config, config, "output_file", "fragments_file")
        config["merge_function"] = merge_function
        if "thresholds" not in config:
            config["thresholds"] = thresholds_lut
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(global_config, config, "find_segments_block_size", "block_size")

    if "FindSegmentsGetChunkedGlobalLUTsTask" in configs:
        config = configs["FindSegmentsGetChunkedGlobalLUTsTask"]
        copy_parameter(input_config, config, "output_file", "fragments_file")
        config["merge_function"] = merge_function
        if "thresholds" not in config:
            config["thresholds"] = thresholds_lut
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(global_config, config, "find_segments_block_size", "block_size")

    if "ExtractSegmentationTask" in configs:
        config = configs["ExtractSegmentationTask"]
        copy_parameter(input_config, config, "output_file", "fragments_file")
        copy_parameter(input_config, config, "output_file", "out_file")
        config["merge_function"] = merge_function
        if "thresholds" not in config:
            config["thresholds"] = thresholds_lut
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(global_config, config, "find_segments_block_size", "block_size")
        copy_parameter(global_config, config, "write_size")

    if "ExtractSuperFragmentSegmentationTask" in configs:
        config = configs["ExtractSuperFragmentSegmentationTask"]
        copy_parameter(input_config, config, "output_file", "fragments_file")
        copy_parameter(input_config, config, "output_file", "out_file")
        config["merge_function"] = merge_function
        copy_parameter(input_config, config, "overwrite")
        copy_parameter(global_config, config, "find_segments_block_size", "block_size")


def copy_parameter(from_config, to_config, name, to_name=None):

    if to_name is None:
        to_name = name
    if name in from_config and to_name not in to_config:
        to_config[to_name] = from_config[name]


def load_skeleton_config(args, aggregate_configs=True):
    if not isinstance(args, list):
        args = [args]
    global_configs = {}
    hierarchy_configs = collections.defaultdict(dict)

    default_configs = load_json_file(args[0]).get("DefaultConfigs", [])

    for default_config_file in default_configs:
        global_configs = merge_dicts(
            load_json_file(default_config_file), global_configs
        )

    for config in args:

        if "=" in config:
            key, val = config.split("=")
            if "." in val:
                try:
                    val = float(val)
                except:
                    pass
            else:
                try:
                    val = int(val)
                except:
                    pass
            if "." in key:
                task, param = key.split(".")
                hierarchy_configs[task][param] = val

        else:
            new_configs = load_json_file(config)
            global_configs = merge_dicts(new_configs, global_configs)
            # keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
            # for k in keys:
            #     if k in global_configs:
            #         if k in new_configs:
            #             global_configs[k].update(new_configs[k])
            #     else:
            #         global_configs[k] = new_configs[k]

            if (
                "Input" in new_configs
                and "config_filename" not in global_configs["Input"]
            ):
                global_configs["Input"]["config_filename"] = config

    # update global confs with hierarchy conf
    for k in hierarchy_configs.keys():
        if k in global_configs:
            global_configs[k].update(hierarchy_configs[k])
        else:
            global_configs[k] = hierarchy_configs[k]

    if aggregate_configs:
        aggregate_configs(global_configs)

    return global_configs


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
