import json
import logging
# import numpy as np
import os
import sys

import daisy
# from segway.tasks import task_helper2 as task_helper
import task_helper2 as task_helper

logger = logging.getLogger(__name__)


class PredictTask(task_helper.SlurmTask):
    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Parameters:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):
        raw_dataset (``string``):
        lsds_file (``string``):
        lsds_dataset (``string``):

            Paths to the input datasets. lsds can be None if not needed.

        out_file (``string``):
        out_dataset (``string``):

            Path to the output datset.

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    train_dir = daisy.Parameter()
    iteration = daisy.Parameter()
    raw_file = daisy.Parameter()
    raw_dataset = daisy.Parameter()
    mask_file = daisy.Parameter(None)
    mask_dataset = daisy.Parameter(None)
    roi_offset = daisy.Parameter(None)
    roi_shape = daisy.Parameter(None)
    roi_context = daisy.Parameter(None)
    out_file = daisy.Parameter()
    out_dataset = daisy.Parameter()
    block_size_in_chunks = daisy.Parameter([1, 1, 1])
    block_size_in_chunks_div = daisy.Parameter([2, 2, 2])
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

    log_to_stdout = daisy.Parameter(default=True)
    log_to_files = daisy.Parameter(default=False)

    num_cores_per_worker = daisy.Parameter(4)
    mem_per_core = daisy.Parameter(2)
    sbatch_gpu_type = daisy.Parameter('any')
    myelin_prediction = daisy.Parameter(0)

    delete_section_list = daisy.Parameter([])
    replace_section_list = daisy.Parameter([])

    overwrite_mask_f = daisy.Parameter(None)
    overwrite_sections = daisy.Parameter(None)

    no_check = daisy.Parameter(False)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        if self.input_shape is not None:
            assert False, "input_shape is deprecated, do not use"
        if self.output_shape is not None:
            assert False, "output_shape is deprecated, do not use"
        if self.out_dtype is not None:
            assert False, "out_dtype is deprecated, do not use"
        if self.out_dims is not None:
            assert False, "out_dims is deprecated, do not use"

        self.setup = os.path.abspath(self.train_dir)
        self.raw_file = os.path.abspath(self.raw_file)
        self.out_file = os.path.abspath(self.out_file)

        logger.info('Input file path: ' + self.raw_file)
        logger.info('Output file path: ' + self.out_file)

        # from here on, all values are in world units (unless explicitly mentioned)

        # get ROI of source
        source = daisy.open_ds(self.raw_file, self.raw_dataset)
        logger.info("Source dataset has shape %s, ROI %s, voxel size %s"%(
            source.shape, source.roi, source.voxel_size))

        # load config
        if os.path.exists(os.path.join(self.setup, 'test_net.json')):
            net_config = json.load(open(os.path.join(self.setup, 'test_net.json')))
            config_file = 'test_net.json'
            meta_file = 'test_net.meta'
        elif os.path.exists(os.path.join(self.setup, 'unet.json')):
            net_config = json.load(open(os.path.join(self.setup, 'unet.json')))
            config_file = 'unet.json'
            meta_file = 'unet.meta'
        elif os.path.exists(os.path.join(self.setup, 'net_io_names.json')):
            raise RuntimeError("net_io_names.json unsupported, please rename network files")
        else:
            raise RuntimeError("No network config found at %s" % self.setup)

        out_dims = net_config['out_dims']
        out_dtype = net_config['out_dtype']
        logger.info('Number of dimensions is %i' % out_dims)

        # get chunk size and context
        voxel_size = source.voxel_size
        self.net_voxel_size = tuple(self.net_voxel_size)
        if self.net_voxel_size != source.voxel_size:

            downsample_factors = daisy.Coordinate(
                    (1, self.xy_downsample, self.xy_downsample))

            assert source.voxel_size*downsample_factors == self.net_voxel_size, (
                    "Source voxel size %s mult by xy_downsample %d does not "
                    "match network voxel size %s" % (
                        source.voxel_size, self.xy_downsample, self.net_voxel_size))
            
            # logger.info("Mismatched net and source voxel size. "
                        # "Assuming downsampling")
            # force same voxel size for net in and output dataset
            voxel_size = self.net_voxel_size

        net_input_size = daisy.Coordinate(net_config["input_shape"])*voxel_size
        net_output_size = daisy.Coordinate(net_config["output_shape"])*voxel_size
        chunk_size = net_output_size
        context = (net_input_size - net_output_size)/2

        # compute sizes of blocks
        # logger.info("block_size_in_chunks: %s" % self.block_size_in_chunks)
        if self.block_size_in_chunks is not None:
            logger.warn("block_size_in_chunks is deprecated")
        self.block_size_in_chunks = [1, 1, 1]
        block_output_size = chunk_size*tuple(self.block_size_in_chunks)
        block_input_size = block_output_size + context*2

        logger.info("Following sizes in world units:")
        logger.info("net input size  = %s" % (net_input_size,))
        logger.info("net output size = %s" % (net_output_size,))
        logger.info("context         = %s" % (context,))
        logger.info("chunk size      = %s" % (chunk_size,))

        # create read and write ROI
        block_read_roi = daisy.Roi((0, 0, 0), block_input_size) - context
        block_write_roi = daisy.Roi((0, 0, 0), block_output_size)

        if self.center_roi_offset:
            assert self.roi_offset is not None, "Cannot use center_roi_offset if roi_offset is None"
            assert self.roi_shape is not None, "Cannot use center_roi_offset if roi_shape is None"
            self.roi_offset = daisy.Coordinate(self.roi_offset)
            self.roi_shape = daisy.Coordinate(self.roi_shape)
            self.roi_offset -= self.roi_shape/2

        if self.roi_context is not None:
            assert self.center_roi_offset is None or self.center_roi_offset == False
            assert self.roi_offset is None
            assert self.roi_shape is None
            self.roi_context = daisy.Coordinate(self.roi_context)
            self.roi_offset = self.roi_context
            self.roi_shape = source.roi.get_shape()
            self.roi_shape -= self.roi_context
            self.roi_shape -= self.roi_context
            # print("self.roi_offset:", self.roi_offset)
            # print("self.roi_shape:", self.roi_shape)
            # logger.info("user context    = %s" % (context,))

        sched_roi, dataset_roi = task_helper.compute_compatible_roi(
                roi_offset=self.roi_offset,
                roi_shape=self.roi_shape,
                sub_roi_offset=self.sub_roi_offset,
                sub_roi_shape=self.sub_roi_shape,
                roi_context=context,
                chunk_size=chunk_size,
                source_roi=source.roi,
            )

        logger.info("Following ROIs in world units:")
        logger.info("Total input ROI  = %s" % sched_roi)
        logger.info("Block read  ROI  = %s" % block_read_roi)
        logger.info("Block write ROI  = %s" % block_write_roi)
        logger.info("Total output ROI = %s" % dataset_roi)

        logging.info('Preparing output dataset')

        write_size = chunk_size
        write_size = write_size / tuple(self.block_size_in_chunks_div)
        logger.info("ZARR write size:")
        logger.info(write_size)

        delete_ds = False
        if self.overwrite:
            delete_ds = True

        self.affs_ds = daisy.prepare_ds(
            self.out_file,
            self.out_dataset,
            dataset_roi,
            voxel_size,
            out_dtype,
            write_size=write_size,
            force_exact_write_size=True,
            num_channels=out_dims,
            compressor={'id': 'zlib', 'level': 3},
            delete=delete_ds,
            )

        if self.myelin_prediction:
            self.myelin_ds = daisy.prepare_ds(
                self.out_file,
                "volumes/myelin",
                dataset_roi,
                voxel_size,
                out_dtype,
                write_size=chunk_size,
                compressor={'id': 'zlib', 'level': 3},
                delete=delete_ds,
                )

        if self.raw_file.endswith('.json'):
            with open(self.raw_file, 'r') as f:
                spec = json.load(f)
                self.raw_file = spec['container']

        self.overwrite_mask = None
        if self.overwrite_mask_f:
            # force precheck = False for any ROI with any voxel in mask = 1
            self.overwrite_mask = daisy.open_ds(
                self.overwrite_mask_f, "overwrite_mask")

        if self.overwrite_sections is not None:
            write_shape = [k for k in dataset_roi.get_shape()]
            write_shape[0] = 40
            write_shape = tuple(write_shape)

            rois = []
            # overwrite_sections_begin = dataset_roi.get_begin()
            for s in self.overwrite_sections:
                write_offset = [k for k in dataset_roi.get_begin()]
                write_offset[0] = s*40
                rois.append(daisy.Roi(write_offset, write_shape))

            self.overwrite_sections = rois

        # print(self.delete_section_list)
        # print(self.replace_section_list)
        # exit(0)

        config = {
            'iteration': int(self.iteration),
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'read_begin': 0,
            'read_size': 0,
            'out_file': self.out_file,
            'out_dataset': self.out_dataset,
            'voxel_size': self.net_voxel_size,
            'train_dir': self.train_dir,
            'write_begin': 0,
            'write_size': 0,
            'xy_downsample': self.xy_downsample,
            'predict_num_core': self.num_cores_per_worker,
            'config_file': config_file,
            'meta_file': meta_file,
            'delete_section_list': self.delete_section_list,
            'replace_section_list': self.replace_section_list,
            'mask_file': self.mask_file,
            'mask_dataset': self.mask_dataset,
        }

        if self.predict_file is not None:
            predict_script = self.predict_file
        else:
            # use the one included in folder
            predict_script = '%s/predict.py' % (self.train_dir)

        # print(predict_script); exit()

        self.sbatch_mem = int(self.num_cores_per_worker*self.mem_per_core)
        if self.sbatch_num_cores is None:
            self.sbatch_num_cores = self.num_cores_per_worker
        self.slurmSetup(config,
                        predict_script,
                        )

        check_function = (
                lambda b: task_helper.check_block(
                    b, self.affs_ds, is_precheck=True, completion_db=self.completion_db, recording_block_done=self.recording_block_done, logger=logger, check_datastore=False, overwrite_sections=self.overwrite_sections),
                lambda b: task_helper.check_block(
                    b, self.affs_ds, is_precheck=False, completion_db=self.completion_db, recording_block_done=self.recording_block_done, logger=logger, overwrite_sections=self.overwrite_sections)
                )

        if self.overwrite:
            check_function = None

        # if self.overwrite_sections is not None:
            # check_function = None

        if self.no_check:
            check_function = (lambda b: False, lambda b: True)


        # any task must call schedule() at the end of prepare
        self.schedule(
            total_roi=sched_roi,
            read_roi=block_read_roi,
            write_roi=block_write_roi,
            # write_size=block_output_size,
            process_function=self.new_actor,
            check_function=check_function,
            read_write_conflict=False,
            fit='overhang',
            num_workers=self.num_workers,
            # log_to_file=True
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
        [{'task': PredictTask(global_config=global_config,
                              **user_configs),
         'request': req_roi}],
        global_config=global_config)
