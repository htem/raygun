# BELOW IS DESIGNED TO ALLOW EASY VIEWING OF ZARR VOLUMES USED BY THE GUNPOWDER PIPELINES
# specifically configured for HTEM servers at Harvard Medical School
import daisy
import zarr
import sys

from funlib.show.neuroglancer import add_layer#, new_layer

from segway.gt_scripts.gt_tools import *

def ng_view_zarr(src, layers=None): #src should be zarr data file (i.e. "/path/to/data/file.zarr")
    if src[-1] == '/':
        src = src[:-1]

    if layers is None:        
        zarr_file = zarr.open(src)
        if 'volumes' in zarr_file.group_keys():
            zarr_file = zarr_file['volumes']
        layers = [array for array in zarr_file.array_keys()]
    
    viewer = make_ng_viewer(unsynced=True)

    with viewer.txn() as s:
        for layer in layers:
            daisy_array = daisy.open_ds(src, 'volumes/'+layer)            
            add_layer(s, [daisy_array], layer)
            print(f'Showing layer {layer}')

    print_ng_link(viewer)
    
    return viewer

if __name__ == "__main__":
    viewer = ng_view_zarr(*sys.argv[1:])
    while True:
        continue
