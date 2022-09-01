from __future__ import print_function

import json
import logging
import os
import glob
import sys

import gpu_utils
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_random_gpu_lowest_memory())
print("Running script on GPU #%s" % os.environ["CUDA_VISIBLE_DEVICES"])

import gunpowder as gp
from gunpowder import Coordinate
# import numpy as np
import pymongo

from synful.gunpowder import IntensityScaleShiftClip


def block_done_callback(
        completion_db,
        block,
        start,
        duration):
    document = dict()
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })
    completion_db.insert(document)


def predict(
        iteration,
        train_dir,
        config_file,
        meta_file,
        raw_file,
        raw_dataset,
        voxel_size,
        xy_downsample,
        out_file,
        predict_num_core,
        db_host,
        db_name,
        completion_db_name,
        # worker_config,
        out_properties,
        **kwargs):

    print("db_host: ", db_host)
    print("db_name: ", db_name)
    print("completion_db_name: ", completion_db_name)
    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    setup_dir = train_dir

    with open(os.path.join(setup_dir, config_file), 'r') as f:
        net_config = json.load(f)

    parameterfile = os.path.join(setup_dir, 'parameter.json')
    if os.path.exists(parameterfile):
        with open(parameterfile, 'r') as f:
            parameters = json.load(f)
    else:
        raise RuntimeError("Untested")
        parameters = {}

    # try to find checkpoint name
    pattern = '*checkpoint_%d.*' % iteration
    checkpoint_files = glob.glob(train_dir + '/' + pattern)
    if len(checkpoint_files) == 0:
        print("Cannot find checkpoints with pattern %s in directory %s" % (
            pattern, train_dir))
        os._exit(1)

    checkpoint_file = checkpoint_files[0].split('.')[0]
    checkpoint_file = checkpoint_file.split('/')[-1]

    # These values are in pixels/voxels
    input_shape = Coordinate(net_config["input_shape"])
    output_shape = Coordinate(net_config["output_shape"])
    voxel_size = Coordinate(tuple(voxel_size))

    context = (input_shape - output_shape)//2

    print("Context is %s" % (context,))
    input_size = input_shape*voxel_size
    output_size = output_shape*voxel_size

    raw = gp.ArrayKey('RAW')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')
    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)

    if 'pred_syn_indicator_out' in out_properties:
        chunk_request.add(pred_post_indicator, output_size)
    if 'pred_partner_vectors' in out_properties:
        chunk_request.add(pred_postpre_vectors, output_size)

    daisy_roi_map = {
        raw: 'read_roi',
    }
    if 'pred_syn_indicator_out' in out_properties:
        daisy_roi_map[pred_post_indicator] = 'write_roi'
    if 'pred_partner_vectors' in out_properties:
        daisy_roi_map[pred_postpre_vectors] = 'write_roi'

    if xy_downsample > 1:
        rawfr = gp.ArrayKey('RAWFR')
        chunk_request.add(rawfr, input_size)

    initial_raw = raw

    if xy_downsample > 1:
        daisy_roi_map[rawfr] = 'read_roi'
        initial_raw = rawfr

    print(out_properties)

    m_property = out_properties[
        'pred_syn_indicator_out'] if 'pred_syn_indicator_out' in out_properties else None
    d_property = out_properties[
        'pred_partner_vectors'] if 'pred_partner_vectors' in out_properties else None

    predict_outputs = {}
    out_datasets = {}
    if 'pred_syn_indicator_out' in out_properties:
        predict_outputs[net_config['pred_syn_indicator_out']] = pred_post_indicator
        out_datasets[pred_post_indicator] = 'volumes/pred_syn_indicator'
    if 'pred_partner_vectors' in out_properties:
        predict_outputs[net_config['pred_partner_vectors']] = pred_postpre_vectors
        out_datasets[pred_postpre_vectors] = 'volumes/pred_partner_vectors'

    # Select Source based on filesuffix.
    if raw_file.endswith('.hdf'):
        pipeline = gp.Hdf5Source(
            raw_file,
            datasets={
                initial_raw: raw_dataset
            },
            array_specs={
                initial_raw: gp.ArraySpec(interpolatable=True),
            }
        )
    elif raw_file.endswith('.zarr'):
        pipeline = gp.ZarrSource(
            raw_file,
            datasets={
                initial_raw: raw_dataset
            },
            array_specs={
                initial_raw: gp.ArraySpec(interpolatable=True),
            }
        )
    elif raw_file.endswith('.n5'):
        pipeline = gp.N5Source(
            raw_file,
            datasets={
                initial_raw: raw_dataset
            },
            array_specs={
                initial_raw: gp.ArraySpec(interpolatable=True),
            }
        )
    else:
        raise RuntimeError('Unknown input data format {}'.format(raw_file))

    pipeline += gp.Pad(initial_raw, size=None)

    if xy_downsample > 1:
        pipeline += gp.DownSample(rawfr, (1, xy_downsample, xy_downsample), raw)

    pipeline += gp.Normalize(raw)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.tensorflow.Predict(
        os.path.join(setup_dir, 'train_net_checkpoint_%d' % iteration),
        inputs={
            net_config['raw']: raw
        },
        outputs=predict_outputs,
        graph=os.path.join(setup_dir, meta_file),
        shared_memory_per_worker_MB=512,
    )

    d_scale = parameters['d_scale'] if 'd_scale' in parameters else None
    if d_scale != 1 and d_scale is not None:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           1. / d_scale,
                                           0)  # Map back to nm world.
    if m_property and 'scale' in m_property:
        if m_property['scale'] != 1:
            pipeline += gp.IntensityScaleShift(pred_post_indicator,
                                               m_property['scale'], 0)
    if d_property is not None and 'scale' in d_property:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors,
                                           d_property['scale'], 0)
    if d_property is not None and 'dtype' in d_property:
        assert d_property['dtype'] == 'int8' or d_property[
            'dtype'] == 'float32', 'predict not adapted to dtype {}'.format(
            d_property['dtype'])
        if d_property['dtype'] == 'int8':
            pipeline += IntensityScaleShiftClip(pred_postpre_vectors,
                                                1, 0, clip=(-128, 127))

    pipeline += gp.ZarrWrite(
        dataset_names=out_datasets,
        output_filename=out_file
    )

    pipeline += gp.PrintProfilingStats(every=100)

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map=daisy_roi_map,
        num_workers=predict_num_core,
        block_done_callback=lambda b, s, d: block_done_callback(
            completion_db,
            b, s, d)
        )

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    db_client.close()
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(
        logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(**run_config)