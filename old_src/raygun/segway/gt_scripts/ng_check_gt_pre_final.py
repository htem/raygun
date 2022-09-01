import daisy
import neuroglancer
import sys

from funlib.show.neuroglancer import add_layer
import gt_tools

arg = sys.argv[1]

if arg.endswith('.json'):
    config = gt_tools.load_config(arg, no_db=True, no_zarr=True)
    f = config["file"]
    raw_file = config["raw_file"]
else:
    raise RuntimeError("Arg is not .json")


if f[-1] == '/':
    f = f[:-1]

viewer = gt_tools.make_ng_viewer()

with viewer.txn() as s:

    add_layer(s, daisy.open_ds(raw_file, 'volumes/raw'), 'raw')
    add_layer(s, daisy.open_ds(f, 'volumes/affs'), 'affs', visible=True)
    add_layer(s, daisy.open_ds(f, 'volumes/fragments'), 'fragments', visible=False)
    add_layer(s, daisy.open_ds(f, config['segment_ds']), 'original_seg', visible=False)
    add_layer(s, daisy.open_ds(f, config['segmentation_skeleton_ds']), 'corrected_seg')
    # add_layer(s, daisy.open_ds(f, 'volumes/segmentation_slice_z_0.100/'), 'zseg_100', visible=False)
    # add_layer(s, daisy.open_ds(f, 'volumes/segmentation_slice_z_0.900/'), 'zseg_900', visible=False)
    add_layer(s, daisy.open_ds(f, config['unlabeled_ds']), 'unlabeled', shader='mask')
    add_layer(s, daisy.open_ds(f, 'volumes/labels/labels_mask_z'), 'mask', shader='mask')
    try:
        add_layer(s, daisy.open_ds(f, 'volumes/labels/unlabeled_mask_skeleton_foreground'), 'unlabeled_foreground', shader='mask')
    except:
        pass


gt_tools.print_ng_link(viewer)
