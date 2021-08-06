import daisy
import sys
import logging
import gt_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
07/29/19
'''

if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)

    if "CatmaidIn" in config:

        in_config = config["CatmaidIn"]
        assert in_config["roi_offset_encoding"] == "tile"
        roi_offset_mul = daisy.Coordinate(in_config["tile_shape"])

    elif "ZarrIn" in config:

        in_config = config["ZarrIn"]
        assert in_config["roi_offset_encoding"] == "nm" or in_config["roi_offset_encoding"] == "voxel"
        roi_offset_mul = daisy.Coordinate((1, 1, 1))

    roi_offset = daisy.Coordinate(in_config["roi_offset"])
    roi_offset = roi_offset * roi_offset_mul

    voxel_size = daisy.Coordinate(in_config["voxel_size"]) 

    if in_config["roi_offset_encoding"] == "nm":
        roi_offset = roi_offset/voxel_size

    print("%d %d %d" % (roi_offset[2], roi_offset[1], roi_offset[0]))
    print("x: %d y: %d z: %d" % (roi_offset[2], roi_offset[1], roi_offset[0]))

    roi_offset_nm = roi_offset * voxel_size
    roi_shape_nm = daisy.Coordinate(in_config["roi_shape_nm"])
    roi_context_nm = daisy.Coordinate(in_config["roi_context_nm"])
    roi_context = roi_context_nm / voxel_size

    print("Grid width: ", roi_shape_nm[2])
    print("Grid height: ", roi_shape_nm[1])

    print("Grid X offset: ", roi_offset_nm[2] % roi_shape_nm[2])
    print("Grid Y offset: ", roi_offset_nm[1] % roi_shape_nm[1])

    print("Z begin: ", roi_offset[0])
    print("Z end: ", roi_offset[0] + int(roi_shape_nm[0]/voxel_size[0]))
    print("Z offset: ", roi_offset[0] - roi_context[0])
