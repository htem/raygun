import json
import logging
import os
import sys

import daisy

from segway.tasks import task_helper2 as task_helper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PredictSynapseTask(task_helper.SlurmTask):

    train_dir = daisy.Parameter()
    iteration = daisy.Parameter()
    raw_file = daisy.Parameter()
    raw_dataset = daisy.Parameter()
    roi_offset = daisy.Parameter(None)
    roi_shape = daisy.Parameter(None)
    out_file = daisy.Parameter()
    # out_dataset = daisy.Parameter()
    out_properties = daisy.Parameter()
    block_size_in_chunks = daisy.Parameter([1, 1, 1])
    num_workers = daisy.Parameter()
    predict_file = daisy.Parameter(None)

    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    center_roi_offset = daisy.Parameter(False)

    # DEPRECATED
    input_shape = daisy.Parameter(None)
    output_shape = daisy.Parameter(None)
    out_dtype = daisy.Parameter(None)
    out_dims = daisy.Parameter(None)

    net_voxel_size = daisy.Parameter(None)
    xy_downsample = daisy.Parameter(1)

    num_cores_per_worker = daisy.Parameter(4)
    mem_per_core = daisy.Parameter(2)
    sbatch_gpu_type = daisy.Parameter('any')
    myelin_prediction = daisy.Parameter(0)

    delete_section_list = daisy.Parameter([])
    replace_section_list = daisy.Parameter([])

    overwrite_mask_f = daisy.Parameter(None)
    overwrite_sections = daisy.Parameter(None)

    no_check = daisy.Parameter(False)

    sched_roi_outside_roi_ok = daisy.Parameter(False)

    def prepare(self):

        self.setup = os.path.abspath(self.train_dir)
        self.raw_file = os.path.abspath(self.raw_file)
        self.out_file = os.path.abspath(self.out_file)

        logger.info('Input file path: ' + self.raw_file)
        logger.info('Output file path: ' + self.out_file)

        # get ROI of source
        source = daisy.open_ds(self.raw_file, self.raw_dataset)
        logger.info("Source dataset has shape %s, ROI %s, voxel size %s"%(
            source.shape, source.roi, source.voxel_size))

        # load config
        if os.path.exists(os.path.join(self.setup, 'test_net_config.json')):
            net_config = json.load(open(os.path.join(self.setup, 'test_net_config.json')))
            config_file = 'test_net_config.json'
            meta_file = 'test_net.meta'
        # elif os.path.exists(os.path.join(self.setup, 'train_net_config.json')):
        #     net_config = json.load(open(os.path.join(self.setup, 'train_net_config.json')))
        #     config_file = 'train_net_config.json'
        #     meta_file = 'train_net.meta'
        # elif os.path.exists(os.path.join(self.setup, 'unet.json')):
        #     net_config = json.load(open(os.path.join(self.setup, 'unet.json')))
        #     config_file = 'unet.json'
        #     meta_file = 'unet.meta'
        else:
            raise RuntimeError("Neither test_net_config.json or unet.json config found at %s" % self.setup)

        # get chunk size and context
        voxel_size = source.voxel_size

        if self.net_voxel_size is None:
            self.net_voxel_size = net_config["voxel_size"]

        self.net_voxel_size = tuple(self.net_voxel_size)
        if self.net_voxel_size != source.voxel_size:
            logger.info("Mismatched net and source voxel size. "
                        "Assuming downsampling")
            assert self.net_voxel_size[1] / source.voxel_size[1] == self.xy_downsample
            assert self.net_voxel_size[2] / source.voxel_size[2] == self.xy_downsample
            # force same voxel size for net in and output dataset
            voxel_size = self.net_voxel_size

        net_input_size = daisy.Coordinate(net_config["input_shape"])*voxel_size
        net_output_size = daisy.Coordinate(net_config["output_shape"])*voxel_size

        chunk_size = net_output_size
        logger.info("block_size_in_chunks: %s" % self.block_size_in_chunks)
        chunk_size = chunk_size * daisy.Coordinate(self.block_size_in_chunks)
        context = (net_input_size - net_output_size)/2

        logger.info("Following sizes in world units:")
        logger.info("net input size  = %s" % (net_input_size,))
        logger.info("net output size = %s" % (net_output_size,))
        logger.info("context         = %s" % (context,))
        logger.info("chunk size      = %s" % (chunk_size,))

        # compute sizes of blocks
        block_output_size = chunk_size
        block_input_size = block_output_size + context*2

        # create read and write ROI
        block_read_roi = daisy.Roi((0, 0, 0), block_input_size) - context
        block_write_roi = daisy.Roi((0, 0, 0), block_output_size)

        input_roi, output_roi = task_helper.compute_compatible_roi(
                roi_offset=self.roi_offset,
                roi_shape=self.roi_shape,
                sub_roi_offset=self.sub_roi_offset,
                sub_roi_shape=self.sub_roi_shape,
                roi_context=context,
                source_roi=source.roi,
                center_roi_offset=self.center_roi_offset,
                chunk_size=chunk_size,
                sched_roi_outside_roi_ok=self.sched_roi_outside_roi_ok
            )

        write_size = chunk_size
        # write_size = write_size / (2, 2, 2)
        logger.info("ZARR write size:")
        logger.info(write_size)

        logging.info('Preparing output dataset')
        print("Preparing output dataset...")
        check_ds = None
        outputs = net_config['outputs']
        delete = True if self.overwrite else False
        for outputname, val in outputs.items():
            out_dims = val['out_dims']
            out_dtype = val['out_dtype']
            scale = None
            if outputname in self.out_properties:
                out_property = self.out_properties[outputname]
                out_dtype = out_property[
                    'dtype'] if 'dtype' in out_property else out_dtype
                scale = out_property['scale'] if 'scale' in out_property else None
                outputname = out_property[
                    'dsname'] if 'dsname' in out_property else outputname
            print('setting dtype to {}'.format(out_dtype))
            out_dataset = 'volumes/%s' % outputname
            print('Number of dimensions is %i' % out_dims)
            ds = daisy.prepare_ds(
                self.out_file,
                out_dataset,
                output_roi,
                voxel_size,
                out_dtype,
                write_size=write_size,
                force_exact_write_size=True,
                num_channels=out_dims,
                compressor={'id': 'gzip', 'level': 5},
                delete=delete
            )
            if scale is not None:
                ds.data.attrs['scale'] = scale
            if check_ds is None:
                check_ds = ds

        if self.raw_file.endswith('.json'):
            with open(self.raw_file, 'r') as f:
                spec = json.load(f)
                self.raw_file = spec['container']

        config = {
            'iteration': self.iteration,
            'train_dir': self.train_dir,
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'voxel_size': voxel_size,
            'xy_downsample': self.xy_downsample,
            'out_file': self.out_file,
            'out_properties': self.out_properties,
            'predict_num_core': self.num_cores_per_worker,
            'config_file': config_file,
            'meta_file': meta_file,
            'delete_section_list': self.delete_section_list,
            'replace_section_list': self.replace_section_list,
        }

        if self.predict_file is not None:
            predict_script = self.predict_file
        else:
            raise RuntimeError("Untested")
            # use the one included in folder
            predict_script = '%s/predict.py' % (self.train_dir)

        self.sbatch_mem = int(self.num_cores_per_worker*self.mem_per_core)
        if self.sbatch_num_cores is None:
            self.sbatch_num_cores = self.num_cores_per_worker
        self.slurmSetup(config,
                        predict_script,
                        )

        check_function = (
                lambda b: task_helper.check_block(
                    b, check_ds, is_precheck=True, completion_db=self.completion_db, recording_block_done=self.recording_block_done, logger=logger, check_datastore=False),
                lambda b: task_helper.check_block(
                    b, check_ds, is_precheck=False, completion_db=self.completion_db, recording_block_done=self.recording_block_done, logger=logger)
                )

        if self.overwrite:
            check_function = None

        if self.no_check:
            check_function = (lambda b: False, lambda b: True)

        # any task must call schedule() at the end of prepare
        self.schedule(
            total_roi=input_roi,
            read_roi=block_read_roi,
            write_roi=block_write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            read_write_conflict=False,
            fit='overhang',
            num_workers=self.num_workers,
            # timeout=self.timeout
            )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': PredictSynapseTask(global_config=global_config,
                              **user_configs),
         'request': req_roi}],
        global_config=global_config)
