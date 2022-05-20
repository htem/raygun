#%% 
import argparse
from functools import partial
import daisy
import skimage.morphology
import numpy as np

def smooth_labels(data, method, radius):
    labels = np.unique(data)
    if hasattr(skimage.morphology, method):
        footprint = skimage.morphology.ball(radius)
        part_method = partial(getattr(skimage.morphology, 'binary_' + method), footprint=footprint)
    else:
        methods = {}
        methods['close'] = partial(smooth_labels, method='closing', radius=radius)
        methods['open'] = partial(smooth_labels, method='opening', radius=radius)
        part_method = lambda x: x
        for this_method in method.split('-'):
            part_method = lambda x: methods[this_method](part_method(x))

    smoothed = np.zeros_like(data)
    for label in labels:
        mask = data == label
        smooth_mask = part_method(mask)
        smoothed[smooth_mask] = label

    return smoothed

#%%
def smooth_segmentation(source_file, source_ds, destination_ds=None, method='closing', radius=2):
    # source_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
    # source_ds = 'volumes/affs0_30nm'
    # destination_ds = 'volumes/affs0min_30nm'

    source = daisy.open_ds(source_file, source_ds)
    destination_file = source_file
    if destination_ds is None:
        destination_ds = f'{source_ds}_{method}_r{radius}'

    chunk_size = 64
    compressor = {  'id': 'blosc', 
                    'clevel': 3,
                    'cname': 'blosclz',
                    'blocksize': chunk_size
                    }
    num_workers = 30
    total_roi = source.roi
    write_size = source.voxel_size * chunk_size
    chunk_roi = daisy.Roi([0,]*len(total_roi.get_offset()), write_size)
    num_channels = 1

    destination = daisy.prepare_ds(
            destination_file,
            destination_ds,
            total_roi,
            source.voxel_size,
            source.dtype,
            write_size=write_size,
            write_roi=chunk_roi,
            num_channels=num_channels,
            compressor=compressor)

    footprint = skimage.morphology.ball(radius)
    smoother = partial(getattr(skimage.morphology, method), footprint=footprint)

    def save_chunk(block:daisy.Roi):
        try:
            destination.__setitem__(block.write_roi, 
                                    smoother(source.to_ndarray(block.read_roi))
                                    )
            return 0 # success
        except:
            return 1 # error
            
    #Write data to new dataset
    success = daisy.run_blockwise(
                total_roi,
                chunk_roi,
                chunk_roi,
                process_function=save_chunk,
                read_write_conflict=False,
                fit='shrink',
                num_workers=num_workers,
                max_retries=2)
    if success:
        print(f'{total_roi} from {source_file}/{source_ds} written to {destination_file}/{destination_ds}')
    else:
        print('Failed to save cutout.')

#%% 
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("source_file", type=str, help='Path to segmentation Zarr/N5.')
    ap.add_argument("source_ds", type=str, help='Name of segmentation layer in the Zarr/N5.')
    ap.add_argument("--destination_ds", type=str, help='Name for the foreground mask layer to output (Default: <source_ds>_<method>_r<radius>).', default=None)
    ap.add_argument("--method", type=str, help='Name of skimage.morphology method to use (options are "closing" or "opening", Default: "closing").', default='closing')
    ap.add_argument("--radius", type=int, help='Radius for sphere used in smoothing (Default: 2).', default=2)
    config = ap.parse_args()

    smooth_segmentation(**vars(config))