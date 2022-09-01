import daisy
import neuroglancer
import sys

from funlib.show.neuroglancer import add_layer
import gt_tools

neuroglancer.set_server_bind_address('0.0.0.0')

config = gt_tools.load_config(sys.argv[1])
f = config["out_file"]
raw_file = config["raw_file"]

if f[-1] == '/':
    f = f[:-1]
if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

raw = daisy.open_ds(raw_file, 'volumes/raw')

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    add_layer(s, raw, 'raw')
    add_layer(s, daisy.open_ds(f, 'volumes/labels/lsd'), 'lsd', shader='rgb')
    add_layer(s, daisy.open_ds(f, 'volumes/labels/labels_mask2'), 'mask', shader='mask')
    # add_layer(s, daisy.open_ds(f, 'volumes/labels/lsd_mask'), 'lsd_mask', shader='mask')
    add_layer(s, daisy.open_ds(f, 'volumes/labels/unlabeled'), 'unlabeled', shader='mask')
    # add_layer(s, daisy.open_ds(f, 'volumes/labels/lsd_unlabeled_mask'), 'lsd_unlabeled', shader='mask')

gt_tools.print_ng_link(viewer)
