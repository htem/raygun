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

    config = gt_tools.load_config(sys.argv[1])

    in_config = config["CatmaidIn"]
    assert in_config["roi_offset_encoding"] == "tile"

    z, y, x = daisy.Coordinate(in_config["roi_offset"]) * daisy.Coordinate(in_config["tile_shape"])
    print("x: %d y: %d z: %d" % (x, y, z))
