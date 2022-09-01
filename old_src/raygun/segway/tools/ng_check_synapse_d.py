import daisy
import neuroglancer
import sys
import numpy as np
import os

from funlib.show.neuroglancer import add_layer
import segway.tasks.task_helper2 as task_helper
from segway.gt_scripts import gt_tools


def open_ds_wrapper(path, ds_name):
    """Returns None if ds_name does not exists """
    try:
        return daisy.open_ds(path, ds_name)
    except KeyError:
        print('dataset %s could not be loaded' % ds_name)
        return None


neuroglancer.set_server_bind_address('0.0.0.0')

try:

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])
    f = global_config["Input"]["output_file"]
    raw_file = global_config["Input"]["raw_file"]

except Exception as e:

    f = sys.argv[1]
    try:
        raw_file = os.environ['raw_file']
    except:
        try:
            # try to guess based on folder directory name
            # assume it's something like xxx/model/iteration/output.zarr
            vol_path = os.path.dirname(os.path.realpath(f))
            print(vol_path)
            index = -4 if vol_path[-1] == '/' else -3
            # index = -4
            vol = vol_path.split('/')[index]
            print(vol)
            if 'pl2' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/segmented_volumes/cb2_pl2_181022.hdf'
            elif 'ml0' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/segmented_volumes/cb2_ml_1204_cleaned.hdf'
            elif 'ml3' in vol:
                raw_file = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/ml3_gt.zarr'
            elif 'ml4' in vol:
                raw_file = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/gt/ml4_gt/ml4_gt.zarr'
            elif 'pl' in vol or 'ml' in vol:
                # pl = 3
                vol = vol.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/%s/cb2_gt_%s.zarr' % (vol, vol)
            elif 'cutout1' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout1/cb2_synapse_cutout1.zarr'
            elif 'cutout2' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout2/cb2_synapse_cutout2.zarr'
            elif 'cutout3' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout3/cb2_synapse_cutout3.zarr'
            elif 'cutout4' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout4/cb2_synapse_cutout4.zarr'
            elif 'cutout5' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout5/cb2_synapse_cutout5.zarr'
            elif 'cutout6' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout6/cb2_synapse_cutout6.zarr'
            elif 'cutout7' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout7/cb2_synapse_cutout7.zarr'
            elif 'cutout8' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout8/cb2_synapse_cutout8.zarr'
            elif 'cutout9' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout9/cb2_synapse_cutout9.zarr'
            elif 'cb2_synapse_cutout' in vol:
                cutout_n = vol[len("cb2_synapse_cutout"):]
                cutout_n = cutout_n.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout%s/cb2_synapse_cutout%s.zarr' % (cutout_n, cutout_n)
            elif 'synapse_cutout' in vol:
                cutout_n = vol[len("synapse_cutout"):]
                cutout_n = cutout_n.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout%s/cb2_synapse_cutout%s.zarr' % (cutout_n, cutout_n)
            elif 'myelin_cutout' in vol:
                cutout_n = vol[len("myelin_cutout"):]
                cutout_n = cutout_n.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/myelin/cutout%s/cb2_myelin_cutout%s.zarr' % (cutout_n, cutout_n)
            else:
                assert(0)
        except:
            assert(0)

synful_f = global_config["synful_output_file"]
synful_f1 = global_config["synful_output_file1"]

if f[-1] == '/':
    f = f[:-1]
if raw_file[-1] == '/':
    raw_file = raw_file[:-1]
if synful_f[-1] == '/':
    synful_f = synful_f[:-1]
if synful_f1[-1] == '/':
    synful_f1 = synful_f1[:-1]


print("Opening %s..." % raw_file)
try:
    raw = daisy.open_ds(raw_file, 'volumes/raw')
except:
    raw = daisy.open_ds(raw_file, 'raw')

print("Opening %s..." % f)
print("Opening %s..." % synful_f)
print("Opening %s..." % synful_f1)

def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader == '255':
        shader="""void main() { emitGrayscale(float(getDataValue().value)); }"""

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


viewer = gt_tools.make_ng_viewer()


pred_dirvector = open_ds_wrapper(synful_f1,
                                 'volumes/pred_partner_vectors')

with viewer.txn() as s:

    add_layer(s, raw, 'raw')

    # add_layer(s, daisy.open_ds(synful_f, 'volumes/pred_syn_indicator'), 'syn_indicator')
    # try: add_layer(s, daisy.open_ds(f, 'volumes/pred_partner_vectors'), 'vector')
    # except: pass

    if pred_dirvector is not None:
        scale_factor = 4  # it is 4 in cb2 prediction
        pred_dirvector.data = np.array(pred_dirvector.data, dtype=np.float32)
        pred_dirvector.data = pred_dirvector.data*4

        nmfactor = 1 if np.max(
            pred_dirvector.data) > 10 else 200  # shader scaling assumed dir vecs in nm
        clipvalue = 100
        # clamp: values are clipped, otherwise not enough signal
        pred_shader = """void main() {{ emitRGB(vec3((
            clamp(getDataValue(0)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
            clamp(getDataValue(1)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f}, (
            clamp(getDataValue(2)*{0}., -{1:.2f}, {1:.2f})+{1:.2f})/{2:.2f})); }}""".format(
            str(nmfactor), clipvalue, clipvalue * 2)
        add_layer(s, pred_dirvector, 'pred_dirvec', shader=pred_shader)

    segment = daisy.open_ds(synful_f, 'volumes/pred_partner_vectors')
    s.navigation.position.voxelCoordinates = np.flip(
        ((segment.roi.get_begin() + segment.roi.get_end()) / 2 / segment.voxel_size))


link = str(viewer)
print(link)
ip_mapping = [
    ['gandalf', 'catmaid3.hms.harvard.edu'],
    ['lee-htem-gpu0', '10.117.28.249'],
    ['lee-lab-gpu1', '10.117.28.82'],
    ['catmaid2', 'catmaid2.hms.harvard.edu'],
    ]
for alias, ip in ip_mapping:
    if alias in link:
        print(link.replace(alias, ip))
