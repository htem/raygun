import daisy
import sys
import json
import os
sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/catpy")

import catpy
from catpy.applications import CatmaidClientApplication
from catpy.applications.export import ExportWidget

import gt_tools


def get_coordinates(offset, shape, context):

    return ((
                (offset[0]-context[0])*4,
                (offset[1]-context[1])*4,
                (offset[2]-context[2])*40,
            ),
            (
                (offset[0]+shape[0]+context[0])*4,
                (offset[1]+shape[1]+context[1])*4,
                (offset[2]+shape[2]+context[2])*40,
            ),
        )


def voxel_to_world(coord):
    return (coord[0]*4, coord[1]*4, coord[2]*40)


def subtract_offset(geometry, offset, context):

    offset = voxel_to_world(offset)
    context = voxel_to_world(context)

    for skid in geometry["skeletons"]:
        for tid in geometry["skeletons"][skid]["treenodes"]:
            for i in range(len(offset)):
                geometry["skeletons"][skid]["treenodes"][tid]["location"][i] -= offset[i]
                geometry["skeletons"][skid]["treenodes"][tid]["location"][i] += context[i]
                # if i == 2:
                #    geometry["skeletons"][skid]["treenodes"][tid]["location"][i] += context[i]

    return geometry


class AnnotationFetcher(CatmaidClientApplication):

    def fetch_all_skeletons(self):
        # https://catmaid.readthedocs.io/en/2018.11.09/api-doc.html#operation---project_id--skeletons--get
        return self.get((self.project_id, 'skeletons'), {})

    def fetch_skeletons_in_bounding_box(self, offset, shape, context):
        '''offset and shape are in pixel'''

        print(get_coordinates(offset, shape, context))
        ((minx, miny, minz), (maxx, maxy, maxz)) = get_coordinates(offset, shape, context)

        # https://catmaid.readthedocs.io/en/stable/api-doc.html#operation---project_id--skeletons-in-bounding-box-get
        return self.get((self.project_id, 'skeletons', 'in-bounding-box'),
                        {
                           'minx': minx,
                           'miny': miny,
                           'minz': minz,
                           'maxx': maxx,
                           'maxy': maxy,
                           'maxz': maxz
                           })


if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)

    force_yes = False
    skeleton_f = None
    if len(sys.argv) > 2:
        if sys.argv[2] == "--yes":
            force_yes = True
        else:
            skeleton_f = sys.argv[2]

    if "skeleton_file" not in config:
        raise RuntimeError('"skeleton_file" not in config')

    if skeleton_f is None:
        skeleton_f = config["skeleton_file"]
        if os.path.exists(skeleton_f) and force_yes is False:
            i = input("Overwrite %s? [y/N] " % skeleton_f)
            if i == '' or i == 'n' or i == 'N':
                print("Aborting...")
                exit(0)

    client = catpy.CatmaidClient(
        # 'http://catmaid3.hms.harvard.edu/catmaidcb2/',
        'http://catmaid.hms.harvard.edu/catmaid3/',
        '14ce27f91d322ab818bb8ccd269f23ff3be1ec6b'  # tringuyen
    )
    project_id = 3  # wfly1 main project ID
    script_name = config["script_name"]

    if "skeleton_url" in config and "catmaid_key" in config:
        client = catpy.CatmaidClient(
            config["skeleton_url"],
            config["catmaid_key"]
        )

    if "skeleton_project_id" in config:
        project_id = config["skeleton_project_id"]
        print("Fetching skeletons for %s" % script_name)
        client.project_id = project_id
        annotation_fetcher = AnnotationFetcher(client)
        skeletons = annotation_fetcher.fetch_all_skeletons()
        export_widget = ExportWidget(client)
        geometry = export_widget.get_treenode_and_connector_geometry(*skeletons)
        with open(skeleton_f, 'w') as f:
            print("Writing to %s" % skeleton_f)
            json.dump(geometry, f)
        exit()

    if "CatmaidIn" in config:
        in_config = config["CatmaidIn"]
        z, y, x = daisy.Coordinate(in_config["roi_offset"]) * daisy.Coordinate(in_config["tile_shape"])
        offset = (x, y, z)
        z, y, x = daisy.Coordinate(in_config["roi_shape_nm"]) / daisy.Coordinate(in_config["voxel_size"])
        shape = (x, y, z)
        z, y, x = daisy.Coordinate(in_config["roi_context_nm"]) / daisy.Coordinate(in_config["voxel_size"])
        context = (x, y, z)

    elif "ZarrIn" in config:

        in_config = config["ZarrIn"]
        voxel_size = daisy.Coordinate(in_config["voxel_size"])
        roi_offset = in_config["roi_offset"]
        if in_config["roi_offset_encoding"] == "voxel":
            roi_offset = [m*n for m, n in zip(roi_offset, voxel_size)]
        elif in_config["roi_offset_encoding"] == "nm":
            pass
        else:
            raise RuntimeError("ZarrIn only supports roi_offset_encoding == `voxel` or `nm`")

        roi_context = daisy.Coordinate(in_config.get("roi_context_nm", [0, 0, 0]))
        roi_shape = daisy.Coordinate(in_config["roi_shape_nm"])
        roi_offset = daisy.Coordinate(roi_offset)
        print("roi_shape: ", roi_shape)
        print("roi_context: ", roi_context)

        if in_config["center_roi_offset"]:
            roi_offset = roi_offset - roi_shape/2

        # roi_shape = roi_shape + roi_context*2
        # roi_offset = roi_offset - roi_context

        roi_context /= voxel_size
        roi_shape /= voxel_size
        roi_offset /= voxel_size

        offset = (roi_offset[2], roi_offset[1], roi_offset[0])
        shape = (roi_shape[2], roi_shape[1], roi_shape[0])
        context = (roi_context[2], roi_context[1], roi_context[0])

    # GT alias: (proj_id, ((offset, shape, context)))
    projects = {
        script_name: (project_id, (offset, shape, context)),
    }

    print(projects)

    pids = [pid for pid in projects]

    for project in pids:
        pid, (offset, shape, context) = projects[project]
        print("Fetching skeletons for %s" % project)

        client.project_id = pid

        annotation_fetcher = AnnotationFetcher(client)
        skeletons = annotation_fetcher.fetch_skeletons_in_bounding_box(offset, shape, context)
        export_widget = ExportWidget(client)
        geometry = export_widget.get_treenode_and_connector_geometry(*skeletons)

        geometry = subtract_offset(geometry, offset, context)

        with open(skeleton_f, 'w') as f:
            print("Writing to %s" % skeleton_f)
            json.dump(geometry, f)
