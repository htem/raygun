#%%
import json
import sys
import daisy
import logging
logging.basicConfig(level=logging.INFO)

sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
from render_CycleGAN import render_tiled

#UTILITY FUNCTIONS
def get_rois(begin=None, center=None, end=None, size=1024, pad=312):
    size = daisy.Coordinate((size,)*3)
    pad = daisy.Coordinate((-pad,)*3)
    if end is not None:
        begin = daisy.Coordinate(end) - size
    elif center is not None:
        begin = daisy.Coordinate(center) - (size // 2)
    elif begin is not None:
        begin = daisy.Coordinate(begin)
    else:
        raise 'Nothing specified...'
    roi = daisy.Roi(begin, size)
    roi_small = roi.grow(pad, pad)

    print('Big ROI:')
    print(f'begin: {roi.get_begin()}')
    print(f'end: {roi.get_end()}')
    
    print('Small ROI:')
    print(f'begin: {roi_small.get_begin()}')
    print(f'end: {roi_small.get_end()}')

    return roi, roi_small

def give_roi(coors):
    coor1, coor2 = [daisy.Coordinate(coor.strip('(').strip(')').split(', ')) for coor in coors.split('\t')]
    return daisy.Roi(coor1, coor2 - coor1)

def show_roi(roi):
    print(f'Coordinate 1: \t{roi.get_begin()}')
    print(f'Coordinate 2: \t{roi.get_end()}')
    print(f"Offset: \t{str(roi.get_offset()).strip('(').strip(')').replace(', ', ',')}")
    print(f"Shape: \t\t{str(roi.get_shape()).strip('(').strip(')').replace(', ', ',')}")
# %%

if __name__ == '__main__':    
    logger = logging.getLogger(__name__)
    logger.info(f'Rendering from config file: {sys.argv[1]}')
    with open(sys.argv[1], 'r') as config_file:
        kwargs = json.load(config_file)
    render_tiled(**kwargs)    