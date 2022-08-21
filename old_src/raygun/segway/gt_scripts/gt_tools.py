import json
import pymongo
# import sys
import os

import socket
import errno

import neuroglancer

# from segway import task_helper
import segway.tasks.task_helper2 as task_helper


def get_db_names(config, file):

    db_name = config.get("db_name", None)
    db_host = config.get("db_host", None)
    db_edges_collection = config.get("db_edges_collection", None)

    if "merge_function" in config:
        db_edges_collection = 'edges_' + config["merge_function"]

    if db_name is None or db_host is None or db_edges_collection is None:

        if "task_config_files" in config:
            configs = config["task_config_files"]
            configs.append("Input.output_file=%s" % file)
            user_configs, global_config = task_helper.parseConfigs(config["task_config_files"])
        else:
            assert "Input" in config
            # print(file); exit(0)
            # print(config["config_f"]); exit(0)
            user_configs, global_config = task_helper.parseConfigs([config["config_f"]])

        if not db_host:
            db_host = global_config["Input"]["db_host"]
        if not db_name:
            db_name = global_config["Input"]["db_name"]
        if not db_edges_collection:
            db_edges_collection = global_config["AgglomerateTask"]["edges_collection"]

    print("db_host: ", db_host)
    print("db_name: ", db_name)
    print("db_edges_collection: ", db_edges_collection)
    # exit(0)
    myclient = pymongo.MongoClient(db_host)
    assert db_name in myclient.database_names(), (
        "db_name %s not found!!!" % db_name)

    return (db_name, db_host, db_edges_collection)


def load_config(config_f, no_db=False, no_zarr=False):

    config_f = config_f.rstrip('/')
    if config_f.endswith(".zarr"):
        config = {}
        config["file"] = config_f
        config["out_file"] = config_f
        config["raw_file"] = config_f

    else:

        with open(config_f) as f:
            config = json.load(f)
            config["config_f"] = config_f

    if "script_name" not in config:
        script_name = os.path.basename(config_f)
        script_name = script_name.split(".")[0]
        config["script_name"] = script_name
    script_name = config["script_name"]

    # try to script_dir the script dir if needed
    script_dir = os.path.split(config_f)[0]

    if "file" in config:
        if config["file"] == "":
            raise RuntimeError('"file" is empty in config...')
    elif "file" in config and config["segment_file"] != "":
        # if config["segment_file"] == "":
        #     raise RuntimeError('"segment_file" is empty in config...')
        config["file"] = config["segment_file"]
    else:
        # assume the following pattern
        file = script_name + '_segmentation_output.zarr'
        if not os.path.exists(file):
            file = script_dir + '/' + file
        config["file"] = file

    file = config["file"]

    if not no_zarr:
        assert os.path.exists(file), "file %s does not exist..." % file

    if not config_f.endswith(".zarr") and not no_db:

        db_name, db_host, db_edges_collection = get_db_names(config, file)
        config["db_name"] = db_name
        config["db_host"] = db_host
        config["db_edges_collection"] = db_edges_collection

    for f in [
            "mask_file",
            "affs_file",
            "fragments_file",
            # "gt_file",
            "segment_file",
            ]:
        if f not in config:
            config[f] = file

    script_name = config["script_name"]
    working_dir = os.path.split(file)[0]
    # script_dir = os.path.split(config_f)[0]

    if "out_file" not in config:
        out_file = working_dir + "/" + script_name + ".zarr"
        config["out_file"] = out_file
    # if not os.path.exists(config["out_file"]):
    #     config["out_file"] = os.path.join(script_dir, config["out_file"])

    if "raw_file" not in config:
        raw_file = working_dir + "/" + script_name + ".zarr"
        config["raw_file"] = raw_file
    if not os.path.exists(config["raw_file"]):
        config["raw_file"] = os.path.join(script_dir, config["raw_file"])

    if "skeleton_file" not in config:
        skeleton_file = working_dir + "/" + script_name + "_skeleton.json"
        config["skeleton_file"] = skeleton_file
    if not os.path.exists(config["skeleton_file"]):
        config["skeleton_file"] = os.path.join(script_dir, config["skeleton_file"])

    if "mask_ds" not in config:
        config["mask_ds"] = "volumes/labels/labels_mask_z"
    if "affs_ds" not in config:
        config["affs_ds"] = "volumes/affs"
    if "fragments_ds" not in config:
        config["fragments_ds"] = "volumes/fragments"
    if "raw_ds" not in config:
        config["raw_ds"] = "volumes/raw"
    if "myelin_ds" not in config:
        config["myelin_ds"] = "volumes/myelin"
    if "segmentation_skeleton_ds" not in config:
        config["segmentation_skeleton_ds"] = "volumes/segmentation_skeleton"
    if "unlabeled_ds" not in config:
        config["unlabeled_ds"] = "volumes/labels/unlabeled_mask_skeleton"
    if "segment_ds_paintera_out" not in config:
        config["segment_ds_paintera_out"] = "volumes/segmentation_paintera"

    return config


def add_ng_layer(s, a, name, shader=None, **kwargs):

    if shader == 'default':
        shader="""void main() { emitGrayscale(toNormalized(getDataValue())); }"""

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    if shader == '255':
        shader="""void main() { emitGrayscale(float(getDataValue().value)); }"""

    if shader == '1':
        shader="""void main() { emitGrayscale(float(getDataValue().value)*float(255)); }"""

    if shader == 'thresh_top50percent':
        shader="""void main() { emitGrayscale(step(0.5, toNormalized(getDataValue()))); }"""

    if shader == 'ramp_top20percent':
        shader="""void main() { emitGrayscale((toNormalized(getDataValue())-float(0.8))*float(5)); }"""

    if shader == 'purple':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue()), 0, toNormalized(getDataValue()))); }"""

    if shader == 'overlay_purple':
        shader="""void main() { emitRGBA(vec4(1, 0, 1, toNormalized(getDataValue()))); }"""

    if shader == 'overlay_purple_ramp_top20percent':
        shader="""void main() { emitRGBA(vec4(1, 0, 1, (toNormalized(getDataValue())-float(0.8))*float(5))); }"""

    if shader == 'overlay_green':
        shader="""void main() { emitRGBA(vec4(0, 1, 0, toNormalized(getDataValue()))); }"""

    #kwargs = {}
    if shader is not None:
       # print(f'Shader: {shader}')
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                offset=a.roi.get_offset()[::-1],
                voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)
    print(s.layers)


def print_ng_link(viewer):

    link = str(viewer)
    print(link)
    ip_mapping = [
        ['gandalf', 'catmaid3.hms.harvard.edu'],
        ['lee-htem-gpu0', '10.117.28.249'],
        ['leelab-gpu-0.med.harvard.edu', '10.11.144.145'],
        ['lee-lab-gpu1', '10.117.28.82'],
        ['catmaid2', 'catmaid2.hms.harvard.edu'],
        ]
    for alias, ip in ip_mapping:
        if alias in link:
            print(link.replace(alias, ip))


def make_ng_viewer(unsynced=False, public=True):

    viewer = None

    if public:
        for i in range(33400, 33500):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(('0.0.0.0', i))
                break
            except socket.error as error:
                if error.errno != errno.EADDRINUSE:
                    raise RuntimeError("Unknown socket error: %s" % (error))
                continue
            finally:
                probe.close()
    else:
        i = 0

    neuroglancer.set_server_bind_address('0.0.0.0', i)
    if unsynced:
        viewer = neuroglancer.UnsynchronizedViewer()
    else:
        viewer = neuroglancer.Viewer()
    if viewer is None:
        raise RuntimeError("Cannot make viewer in port range 33400-33500")

    return viewer

# def new_viewer():
#     for i in range(33400, 33500):
#         probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         try:
#             probe.bind(('0.0.0.0', i))
#             break
#         except OSError as err:
#             if err.errno == 98:
#                 # Address already in use
#                 continue
#             else:
#                 raise err
#         finally:
#             probe.close()

#     print(i)
#     neuroglancer.set_server_bind_address('0.0.0.0', i)
#     viewer = neuroglancer.Viewer()
#     return viewer
