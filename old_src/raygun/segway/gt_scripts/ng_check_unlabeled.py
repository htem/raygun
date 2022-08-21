import daisy
import neuroglancer
import sys

import gt_tools

config = gt_tools.load_config(sys.argv[1])
f = config["file"]
raw_file = config["raw_file"]

if f[-1] == '/':
    f = f[:-1]
if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

raw = daisy.open_ds(raw_file, 'volumes/raw')

viewer = gt_tools.make_ng_viewer()

with viewer.txn() as s:

    gt_tools.add_ng_layer(s, raw, 'raw')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, config['segmentation_skeleton_ds']), 'corrected_segmentation')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, config['unlabeled_ds']), 'unlabeled', shader='1')

    # add(s, daisy.open_ds(f, 'volumes/affs'), 'aff', shader='rgb')
    # # add(s, daisy.open_ds(f, 'volumes/fragments'), 'frag')
    # add(s, daisy.open_ds(f, 'volumes/segmentation_skeleton'), 'seg_skel')
    # # add(s, daisy.open_ds(f, 'volumes/segmentation_skeleton_900'), 'seg_skel')
    # # add(s, daisy.open_ds(f, 'volumes/labels/unlabeled_mask_skeleton'), 'unlabeled', shader='255')

    # # add(s, daisy.open_ds(f, 'volumes/labels/neuron_ids_myelin'), 'ids_myelin')
    # # add(s, daisy.open_ds(f, 'volumes/labels/labels_mask_z'), 'labels_mask_z', shader='255')
    # #add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), 'seg')

    # add(s, daisy.open_ds(f, 'volumes/segmentation_0.800'), 'seg_700')
    # add(s, daisy.open_ds(f, 'volumes/segmentation_0.800'), 'seg_800')
    # # add(s, daisy.open_ds(f, 'volumes/segmentation_0.900'), 'seg_900')

    # # add(s, ul, 'unlabeled', shader='255')

gt_tools.print_ng_link(viewer)
