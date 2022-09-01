import json
import os
import logging
import numpy as np
import sys
import daisy
import shutil
from datetime import datetime

from collections import OrderedDict
# from PIL import Image

import ilastik_main
from ilastik.applets.dataSelection import DatasetInfo
from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
from ilastik.workflows import AutocontextTwoStage
import vigra

logging.basicConfig(level=logging.INFO)


def downsample_data(data, factor):
    slices = tuple(
        slice(None, None, k)
        for k in factor)
    return data[slices]


def replace_sections(array, replace_section_list):

    z_begin = int(array.roi.get_begin()[0] / array.voxel_size[0])
    z_end = int(array.roi.get_end()[0] / array.voxel_size[0])

    for z, z_replace in replace_section_list:

        if ((z >= z_begin and z < z_end) and
                (z_replace >= z_begin and z_replace < z_end)):
            z -= z_begin
            z_replace -= z_begin
            array.data[z] = array.data[z_replace]


def predict_2d(
        block,
        raw_ds,
        output_ds,
        network_voxel_size,
        clamp_threshold,
        replace_section_list
        ):

    raw_ndarray = raw_ds[block.read_roi].to_ndarray()
    # we need to slice 3D volume to 2D and downsample it
    # downsample_factors = (1, network_voxel_size, network_voxel_size)
    downsample_factors = network_voxel_size / raw_ds.voxel_size
    print(downsample_factors)
    raw_ndarray = downsample_data(raw_ndarray, downsample_factors)

    inputs = []
    for raw in raw_ndarray:
        input_data = vigra.taggedView(raw, 'yx')
        inputs.append(DatasetInfo(preloaded_array=input_data))

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    role_data_dict = OrderedDict(
          [("Raw Data", inputs)])

    # Run the export via the BatchProcessingApplet
    predictions = shell.workflow.batchProcessingApplet.run_export(
        role_data_dict, export_to_array=True)

    # get only labels
    labels = []
    for o in predictions:
        labels.append(o[:, :, 0])
    # 2d arrays to 3d array
    out_ndarray = np.stack(labels)
    # convert float to uint8
    out_ndarray = np.stack(out_ndarray)
    out_ndarray = np.array((1-out_ndarray)*255, dtype=np.uint8)

    if clamp_threshold:
        # out_ndarray = np.place(out_ndarray, out_ndarray < clamp_threshold, 0)
        out_ndarray[out_ndarray < clamp_threshold] = 0
        out_ndarray[out_ndarray >= clamp_threshold] = 255

    # write only write_roi
    out_array = daisy.Array(out_ndarray, block.read_roi, network_voxel_size)
    replace_sections(out_array, replace_section_list)
    output_ds[block.write_roi] = out_array[block.write_roi]


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    raw_file = None
    raw_dataset = None
    out_file = None
    out_dataset = None
    network_voxel_size = None
    lazyflow_num_threads = None
    lazyflow_mem = None
    ilastik_project_path = None
    clamp_threshold = 0
    replace_section_list = []

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    if run_config.get('block_id_add_one_fix', False):
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    logging.info("Reading raw from %s", raw_file)
    raw_ds = daisy.open_ds(raw_file, raw_dataset, mode='r')
    output_ds = daisy.open_ds(out_file, out_dataset, mode="r+")
    assert output_ds.data.dtype == np.uint8

    assert network_voxel_size is not None, "Untested"
    network_voxel_size = daisy.Coordinate(network_voxel_size)

    print(replace_section_list)

    # make a local copy of the ilastik project because the framework doesn't allow
    # for concurrent open across jobs/workers

    print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
    while True:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
        local_ilastik_project_path = "%s_%s.ilp" % (config_file, timestamp)
        if os.path.exists(local_ilastik_project_path):
            continue
        shutil.copyfile(ilastik_project_path, local_ilastik_project_path)
        break

    client_scheduler = daisy.Client()

    os.environ["LAZYFLOW_THREADS"] = str(lazyflow_num_threads)
    os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(lazyflow_mem)

    args = ilastik_main.parser.parse_args([])
    args.headless = True
    args.project = local_ilastik_project_path

    shell = ilastik_main.main(args)
    assert isinstance(shell.workflow,
                      (PixelClassificationWorkflow, AutocontextTwoStage))

    # Obtain the training operator
    opPixelClassification = shell.workflow.pcApplet.topLevelOperator

    # Sanity checks
    assert len(opPixelClassification.InputImages) > 0
    assert opPixelClassification.Classifier.ready()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        predict_2d(
            block,
            raw_ds,
            output_ds,
            network_voxel_size,
            clamp_threshold,
            replace_section_list)

        client_scheduler.release_block(block, ret=0)

    # remove tmp binary
    os.remove(local_ilastik_project_path)
