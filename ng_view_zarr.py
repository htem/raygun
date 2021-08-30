# BELOW IS DESIGNED TO ALLOW EASY VIEWING OF ZARR VOLUMES USED BY THE GUNPOWDER PIPELINES
# specifically configured for HTEM servers at Harvard Medical School
import daisy
import neuroglancer
import zarr

def ng_view_zarr(src, layers=None):
    neuroglancer.set_server_bind_address('0.0.0.0')
    # config = gt_tools.load_config(sys.argv[1])
    if src[-1] == '/':
        src = src[:-1]

    zarr_file = daisy.open_ds(src, 'volumes/raw')

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add_layer(s, zarr_file, 'raw')
        # xyz
        # s.navigation.position.voxelCoordinates = (63*1024, 180*1024, 1129)

    link = str(viewer)
    print(link)
    ip_mapping = [
        ['gandalf', 'catmaid3.hms.harvard.edu'],
        ['lee-htem-gpu0', '10.117.28.249'],
        ['lee-lab-gpu1', '10.117.28.82'],
        ]
    for alias, ip in ip_mapping:
        if alias in link:
            print(link.replace(alias, ip))


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

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                offset=a.roi.get_offset()[::-1],
                voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)
    print(s.layers)