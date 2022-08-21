import daisy
import neuroglancer
import sys

import gt_tools

neuroglancer.set_server_bind_address('0.0.0.0')

config = gt_tools.load_config(sys.argv[1])
f = config["file"]
raw_file = config["raw_file"]

if f[-1] == '/':
    f = f[:-1]
if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

raw = daisy.open_ds(raw_file, 'volumes/raw')

# viewer = neuroglancer.Viewer()
viewer = gt_tools.make_ng_viewer(public=True)

with viewer.txn() as s:

    gt_tools.add_ng_layer(s, raw, 'raw')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, config['segment_ds']), 'seg')

gt_tools.print_ng_link(viewer)
