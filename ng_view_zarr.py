# BELOW IS DESIGNED TO ALLOW EASY VIEWING OF ZARR VOLUMES USED BY THE GUNPOWDER PIPELINES
# specifically configured for HTEM servers at Harvard Medical School
import daisy
import neuroglancer
import zarr
import sys

from segway.gt_scripts.gt_tools import *

def ng_view_zarr(src, layers=None): #src should be zarr data file (i.e. "/path/to/data/file.zarr")
    if src[-1] == '/':
        src = src[:-1]

    if layers is None:        
        zarr_file = zarr.open(src+'/volumes')
        layers = [array for array in zarr_file.array_keys()]
    
    viewer = make_ng_viewer(unsynced=True)

    with viewer.txn() as s:
        for layer in layers:
            daisy_array = daisy.open_ds(src, 'volumes/'+layer)            
            add_layer(s, daisy_array, layer)
        # xyz
        # s.navigation.position.voxelCoordinates = (63*1024, 180*1024, 1129)

    print_ng_link(viewer)
    
    return viewer


def add_layer(s, a, name, shader=None):
    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    if shader == '255':
        shader="""void main() { emitGrayscale(float(getDataValue().value)); }"""

    if shader == '1':
        shader="""void main() { emitGrayscale(float(getDataValue().value)*float(255)); }"""

    kwargs = {}
    if shader is not None:
        kwargs['shader'] = shader
    
    rank = a.roi.dims()
    dimensions = neuroglancer.CoordinateSpace(
                names=['z', 'y', 'x'],
                units=['nm'] * rank,
                scales=list(a.voxel_size),
            )

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                voxel_offset=a.roi.get_offset()[::-1],
                dimensions=dimensions
                # voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)
    # print(s.layers)

if __name__ == "__main__":
    # print(*sys.argv)
    viewer = ng_view_zarr(*sys.argv[1:])
    while True:
        x = 1