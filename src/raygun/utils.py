from io import StringIO
from jsmin import jsmin
import json
import os
import numpy as np
import gunpowder as gp


def passing_locals(local_dict):
    kwargs = {}
    for k, v in local_dict.items():
        if k[0] != "_" and k != "self":
            if k == "kwargs":
                kwargs.update(v)
            else:
                kwargs[k] = v
    return kwargs


def get_config_name(config_path, base_folder):
    config_name = os.path.dirname(config_path)
    config_name = config_name.replace(base_folder, "")
    config_name = "_".join(config_name.split("/"))[1:]

    return config_name


def calc_max_padding(
    output_size, voxel_size, neighborhood=None, sigma=None, mode="shrink"
):

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = gp.Coordinate(
            [np.abs(aff) for val in neighborhood for aff in val if aff != 0]
        )

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


def serialize(obj):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out[key] = serialize(value)
        return out
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def to_json(obj, file):
    out = serialize(obj)
    with open(file, "w") as f:
        json.dump(out, f)


def load_json_file(fin):
    with open(fin, "r") as f:
        config = json.load(StringIO(jsmin(f.read())))
    return config


def merge_dicts(from_dict, to_dict):

    # merge first level
    for k in from_dict:
        if k not in to_dict:
            to_dict[k] = from_dict[k]
        else:
            # overwrite merge second level
            for kk in from_dict[k]:
                to_dict[k][kk] = from_dict[k][kk]

    return to_dict
