import daisy
import neuroglancer
import sys
import os
import gt_tools

dataset = None

if '.zarr' in sys.argv[1]:
    raw_file = sys.argv[1]
else:
    config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)
    raw_file = config["raw_file"]
    dataset = config.get('raw_dataset', None)

if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

if dataset:
   raw = daisy.open_ds(raw_file, dataset)
else:
    try:
       raw = daisy.open_ds(raw_file, 'volumes/raw')
    except:
       raw = daisy.open_ds(raw_file, 'raw')

viewer = gt_tools.make_ng_viewer()

with viewer.txn() as s:

    gt_tools.add_ng_layer(s, raw, 'raw')

print("Raw ZARR at %s" % os.path.realpath(raw_file))

gt_tools.print_ng_link(viewer)
