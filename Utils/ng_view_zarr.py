# BELOW IS DESIGNED TO ALLOW EASY VIEWING OF ZARR VOLUMES USED BY THE GUNPOWDER PIPELINES
# specifically configured for HTEM servers at Harvard Medical School
import daisy
import zarr
import sys
import neuroglancer
import socket
import errno

from funlib.show.neuroglancer import add_layer#, new_layer

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

def get_specs(layer):
    s_start = layer.lower().find('seed')
    s_end = layer[s_start:].find('_')
    seed_num = layer[s_start+4:s_start+s_end]

    c_start = layer.lower().find('checkpoint')
    c_end = layer[c_start:].find('_')
    check_num = layer[c_start+10:c_start+c_end]

    label = f's{seed_num}_c{check_num}'

    if 'net' in layer.lower().split('_')[-1]:
        label += f'_{layer.split("_")[-1]}'

    return label

def get_label(layer):
    if 'split' in layer.lower():
        return 'Split_' + get_specs(layer)
    elif 'link' in layer.lower():
        return 'Link_' + get_specs(layer)
    else:
        return layer

def ng_view_zarr(src, layers=None): #src should be zarr data file (i.e. "/path/to/data/file.zarr")
    if src[-1] == '/':
        src = src[:-1]

    if layers is None:        
        zarr_file = zarr.open(src)
        if 'volumes' in zarr_file.group_keys():
            zarr_file = zarr_file['volumes']
            prepend = 'volumes/'
        else:
            prepend = ''
        layers = [array for array in zarr_file.array_keys()]
        groups = [group for group in zarr_file.group_keys()]
        for group in groups:
            layers.append([array for array in zarr_file[group].array_keys()])
    else:
        prepend = ''
    
    viewer = make_ng_viewer(unsynced=True)

    with viewer.txn() as s:
        g = 0
        for layer in layers:
            if type(layer) == list:
                daisy_arrays = [daisy.open_ds(src, prepend+groups[g]+'/'+level) for level in layer]         
                add_layer(s, daisy_arrays, groups[g])                
                print(f'Showing layer {groups[g]}')
                g += 1
            else:
                daisy_array = daisy.open_ds(src, prepend+layer)            
                label = get_label(layer)
                add_layer(s, [daisy_array], label)
                print(f'Showing layer {label}')

    print_ng_link(viewer)
    
    return viewer

if __name__ == "__main__":
    viewer = ng_view_zarr(*sys.argv[1:])
    while True:
        continue
