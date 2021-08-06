import copy
import datetime
import json
import logging
import multiprocessing
import hashlib
import subprocess
import os
import collections
import pymongo
import numpy as np
from io import StringIO
from jsmin import jsmin

import daisy
import ast

logger = logging.getLogger(__name__)


class SlurmTask(daisy.Task):

    max_retries = daisy.Parameter(2)

    log_dir = daisy.Parameter()

    sbatch_num_cores = daisy.Parameter(1)
    sbatch_time = daisy.Parameter("1:00:00")
    sbatch_mem = daisy.Parameter(4)
    sbatch_partition = daisy.Parameter(None)
    sbatch_account = daisy.Parameter(None)
    sbatch_gpu_type  = daisy.Parameter(None)

    debug_print_command_only = daisy.Parameter(False)
    overwrite = daisy.Parameter(False)
    no_check_dependency = daisy.Parameter(False)
    no_precheck = daisy.Parameter(False)

    db_host = daisy.Parameter()
    db_name = daisy.Parameter()

    num_workers = daisy.Parameter(1)

    timeout = daisy.Parameter(None)
    completion_db_class_name = daisy.Parameter(None)

    def slurmSetup(
            self, config, actor_script,
            python_module=False,
            python_interpreter='python',
            completion_db_name_extra=None,
            **kwargs):
        '''Write config file and sbatch file for the actor, and generate
        `new_actor_cmd`. We also keep track of new jobs so to kill them
        when the task is finished.'''

        if not python_module:
            logname = (actor_script.split('.'))[-2].split('/')[-1]
        else:
            logname = (actor_script.split('.'))[-1]

        for prepend in [
                '.',
                '/n/groups/htem/Segmentation/shared-nondev',
                os.path.dirname(os.path.realpath(__file__))
                ]:
            if os.path.exists(os.path.join(prepend, actor_script)):
                actor_script = os.path.realpath(os.path.join(prepend, actor_script))
                actor_script = actor_script.replace('/mnt/orchestra_nfs', '/n/groups/htem')
                break
        else:
            logname = (actor_script.split('.'))[-1]

        f = "%s/%s.error_blocks.%s" % (self.log_dir, logname, str(datetime.datetime.now()).replace(' ', '_'))
        self.error_log = open(f, "w")
        self.precheck_log = None

        db_client = pymongo.MongoClient(self.db_host)
        db = db_client[self.db_name]

        if self.completion_db_class_name:
            class_name = self.completion_db_class_name
        else:
            class_name = self.__class__.__name__
        completion_db_name = class_name + '_fb'
        if completion_db_name_extra:
            completion_db_name = completion_db_name + completion_db_name_extra

        if completion_db_name not in db.list_collection_names():
            self.completion_db = db[completion_db_name]
            self.completion_db.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
        else:
            self.completion_db = db[completion_db_name]

        config.update({
            'db_host': self.db_host,
            'db_name': self.db_name,
            'completion_db_name': completion_db_name,
            })

        self.slurmtask_run_cmd, self.new_actor_cmd = generateActorSbatch(
            config,
            actor_script,
            python_module=python_module,
            python_interpreter=python_interpreter,
            log_dir=self.log_dir,
            logname=logname,
            sbatch_num_cores=self.sbatch_num_cores,
            sbatch_time=self.sbatch_time,
            sbatch_mem=self.sbatch_mem,
            sbatch_gpu_type=self.sbatch_gpu_type,
            sbatch_partition=self.sbatch_partition,
            sbatch_account=self.sbatch_account,
            **kwargs)
        self.started_jobs = multiprocessing.Manager().list()
        self.started_jobs_local = []

        self.logname = logname
        self.task_done = False
        self.launch_process_cmd = multiprocessing.Manager().dict()

        self.shared_precheck_blocks = multiprocessing.Manager().list()
        self.shared_error_blocks = multiprocessing.Manager().list()


    def new_actor(self):
        '''Submit new actor job using sbatch'''
        context_str = os.environ['DAISY_CONTEXT']

        logger.info("Srun command: DAISY_CONTEXT={} CUDA_VISIBLE_DEVICES=0 {}".format(
                context_str,
                self.slurmtask_run_cmd))

        logger.info("Submit command: DAISY_CONTEXT={} {}".format(
                context_str,
                ' '.join(self.new_actor_cmd)))

        run_cmd = "cd %s" % os.getcwd() + "; "
        run_cmd += "DAISY_CONTEXT=%s" % context_str + " "
        run_cmd += ' '.join(self.new_actor_cmd)

        process_cmd = run_cmd

        print(process_cmd)
        self.launch_process_cmd['cmd'] = process_cmd
        if not self.debug_print_command_only:

                worker_id = daisy.Context.from_env().worker_id
                logout = open("%s/%s.%d.out" % (
                                        self.log_dir, self.logname, worker_id),
                              'a')
                logerr = open("%s/%s.%d.err" % (
                                        self.log_dir, self.logname, worker_id),
                              'a')
                cp = subprocess.run(process_cmd,
                                    stdout=logout,
                                    stderr=logerr,
                                    shell=True
                                    )

    def cleanup(self):

        if self.error_log:
            for b in self.shared_error_blocks:
                self.error_log.write(str(b) + '\n')
            self.error_log.close()

        if self.precheck_log:
            for b in self.shared_precheck_blocks:
                self.precheck_log.write(str(b) + '\n')
            self.precheck_log.close()

        try:
            started_slurm_jobs = self.started_jobs._getvalue()
        except:
            try:
                started_slurm_jobs = self.started_jobs_local
            except:
                started_slurm_jobs = []

        self.task_done = True

    def _periodic_callback(self):
        try:
            self.started_jobs_local = self.started_jobs._getvalue()
        except:
            pass

        if not self.task_done:
            if self.launch_process_cmd is not None and 'cmd' in self.launch_process_cmd:
                print("Launch command: ", self.launch_process_cmd['cmd'])


    def log_error_block(self, block):

        self.error_log.write(str(block) + '\n')

    def recording_block_done(self, block):

        document = dict()
        document.update({
            'block_id': block.block_id,
            'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
            'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
            'start': 0,
            'duration': 0
        })
        self.completion_db.insert(document)


def generateActorSbatch(
        config, actor_script,
        python_module,
        log_dir, logname,
        python_interpreter,
        **kwargs):

    config_str = ''.join(['%s' % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))
    try:
        os.makedirs('.run_configs')
    except Exception:
        pass
    config_file = os.path.join(
        '.run_configs', '%s_%d.config' % (logname, config_hash))
    with open(config_file, 'w') as f:
        json.dump(config, f)

    if not python_module:
        run_cmd = ' '.join([
            python_interpreter,
            '%s' % actor_script,
            '%s' % config_file,
            ])
    else:
        run_cmd = ' '.join([
            python_interpreter,
            '-m',
            '%s' % actor_script,
            '%s' % config_file,
            ])

    sbatch_script = os.path.join('.run_configs', '%s_%d.sh'%(logname, config_hash))
    generateSbatchScript(
        sbatch_script, run_cmd, log_dir, logname,
        **kwargs)

    new_actor_cmd = [
        'sh',
        '%s' % sbatch_script
        ]

    return run_cmd, new_actor_cmd


def generateSbatchScript(
        sbatch_script,
        run_cmd,
        log_dir,
        logname,
        sbatch_time="1:00:00",
        sbatch_num_cores=1,
        sbatch_mem=6,
        sbatch_gpu_type=None,
        sbatch_partition=None,
        sbatch_account=None,
        ):
    text = []
    text.append("#!/bin/bash")
    text.append("#SBATCH -t %s" % sbatch_time)

    if sbatch_gpu_type is not None:
        if sbatch_partition is None:
            sbatch_partition = 'gpu'
        if sbatch_gpu_type == '' or sbatch_gpu_type == 'any':
            text.append("#SBATCH --gres=gpu:1")
        else:
            text.append("#SBATCH --gres=gpu:{}:1".format(sbatch_gpu_type))

    if sbatch_partition is None:
        sbatch_partition = 'short'
    text.append("#SBATCH -p %s" % sbatch_partition)

    if sbatch_account:
        text.append("#SBATCH --account %s" % sbatch_account)

    text.append("#SBATCH -c %d" % sbatch_num_cores)
    text.append("#SBATCH --mem=%dGB" % sbatch_mem)
    text.append("#SBATCH -o {}/{}_%j.out".format(log_dir, logname))
    text.append("#SBATCH -e {}/{}_%j.err".format(log_dir, logname))
    # text.append("#SBATCH -o .logs_sbatch/{}_%j.out".format(logname))
    # text.append("#SBATCH -e .logs_sbatch/{}_%j.err".format(logname))

    text.append("")
    # text.append("$*")
    text.append(run_cmd)

    logger.info("Writing sbatch script %s" % sbatch_script)
    with open(sbatch_script, 'w') as f:
        f.write('\n'.join(text))


def parseConfigs(args, aggregate_configs=True):
    global_configs = {}
    user_configs = {}
    hierarchy_configs = collections.defaultdict(dict)

    # first load default configs if avail
    try:
        config_file = "segway/tasks/task_defaults.json"
        with open(config_file, 'r') as f:
            # global_configs = {**json.load(f), **global_configs}
            global_configs = {**json.load(StringIO(jsmin(f.read()))), **global_configs}
    except Exception:
        try:
            config_file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/segway/tasks/task_defaults.json"
            with open(config_file, 'r') as f:
                # global_configs = {**json.load(f), **global_configs}
                global_configs = {**json.load(StringIO(jsmin(f.read()))), **global_configs}
        except:
            logger.info("Default task config not loaded")

    for config in args:

        if "=" in config:
            key, val = config.split('=')
            if "." in val:
                try: val = float(val)
                except: pass
            else:
                try: val = int(val)
                except: pass
            if '.' in key:
                task, param = key.split('.')
                hierarchy_configs[task][param] = val
            else:
                user_configs[key] = ast.literal_eval(val)
        else:
            with open(config, 'r') as f:
                # print("helper: loading %s" % config)
                # new_configs = json.load(f)
                new_configs = json.load(StringIO(jsmin(f.read())))
                print(new_configs)
                keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
                for k in keys:
                    if k in global_configs:
                        if k in new_configs:
                            global_configs[k].update(new_configs[k])
                    else:
                        global_configs[k] = new_configs[k]
                # print(list(global_configs.keys()))

                if 'Input' in new_configs and 'config_filename' not in global_configs['Input']:
                    global_configs['Input']['config_filename'] = config

    # print("\nhelper: final config")
    # print(global_configs)

    # update global confs with hierarchy conf
    # print(hierarchy_configs)
    for k in hierarchy_configs.keys():
        if k in global_configs:
            global_configs[k].update(hierarchy_configs[k])
        else:
            global_configs[k] = hierarchy_configs[k]
    # global_configs = {**global_configs, **hierarchy_configs}

    if aggregate_configs:
        aggregateConfigs(global_configs)
    return (user_configs, global_configs)


def copyParameter(from_config, to_config, name, to_name=None):

    if to_name is None:
        to_name = name
    if name in from_config and to_name not in to_config:
        to_config[to_name] = from_config[name]


def aggregateConfigs(configs):

    input_config = configs["Input"]
    network_config = configs["Network"]
    synapse_network_config = configs["SynfulNetwork"]

    today = datetime.date.today()
    parameters = {}
    parameters['year'] = today.year
    parameters['month'] = '%02d' % today.month
    parameters['day'] = '%02d' % today.day
    parameters['network'] = network_config['name']
    parameters['synful_network'] = synapse_network_config['name']
    parameters['synful_network1'] = synapse_network_config['name1']
    parameters['iteration'] = network_config['iteration']
    parameters['synful_iteration'] = synapse_network_config['iteration']
    parameters['synful_iteration1'] = synapse_network_config['iteration1']
    config_filename = input_config['config_filename']

    parameters['proj'] = input_config.get('proj', '')
    if parameters['proj'] == '':
        # proj is just the last folder in the config path
        parameters['proj'] = config_filename.split('/')[-2]

    script_name = config_filename.split('/')[-1].split('.')
    if len(script_name) > 2:
        raise RuntimeError("script_name name %s cannot have more than two `.`")
    else:
        script_name = script_name[0]
    parameters['script_name'] = script_name
    parameters['script_folder'] = parameters['proj']
    parameters['script_dir'] = '/'.join(config_filename.split('/')[0:-1])
    script_dir = parameters['script_dir']

    input_config["experiment"] = input_config["experiment"].format(**parameters)
    parameters['experiment'] = input_config["experiment"]

    # input_config["output_file"] = input_config["output_file"].format(**parameters)

    input_config_synful = copy.deepcopy(input_config)
    input_config_synful1 = copy.deepcopy(input_config)
    parameters_synful = copy.deepcopy(parameters)
    parameters_synful['network'] = parameters_synful['synful_network']
    parameters_synful['iteration'] = parameters_synful['synful_iteration']
    parameters_synful1 = copy.deepcopy(parameters)
    parameters_synful1['network'] = parameters_synful1['synful_network1']
    parameters_synful1['iteration'] = parameters_synful1['synful_iteration1']

    for config in input_config:
        if isinstance(input_config[config], str):
            input_config[config] = input_config[config].format(**parameters)
    # print(input_config_synful); exit()
    # print(parameters_synful); exit()
    for config in input_config_synful:
        if isinstance(input_config_synful[config], str):
            input_config_synful[config] = input_config_synful[config].format(**parameters_synful)
    # print(input_config_synful); exit()
    for config in input_config_synful1:
        if isinstance(input_config_synful1[config], str):
            input_config_synful1[config] = input_config_synful1[config].format(**parameters_synful1)

    configs["output_file"] = input_config["output_file"]
    configs["synful_output_file"] = input_config_synful["output_file"]
    configs["synful_output_file1"] = input_config_synful1["output_file"]

    for path_name in ["output_file", "synful_output_file", "synful_output_file1"]:

        output_path = configs[path_name]
        if not os.path.exists(output_path):
            output_path = os.path.join(script_dir, output_path)
        output_path = os.path.abspath(output_path)
        if output_path.startswith("/mnt/orchestra_nfs/"):
            output_path = output_path[len("/mnt/orchestra_nfs/"):]
            output_path = "/n/groups/htem/" + output_path

    os.makedirs(input_config['log_dir'], exist_ok=True)

    merge_function = configs["AgglomerateTask"]["merge_function"]
    thresholds = configs["ExtractSegmentationFromLUTBlockwiseTask"]["thresholds"]
    thresholds_lut = thresholds
    try:
        thresholds_lut = configs["FindSegmentsBlockwiseTask4"]["thresholds"]
    except: pass

    for config in configs:

        if "Task" not in config:
            # print("Skipping %s" % config)
            continue

        config = configs[config]
        copyParameter(input_config, config, 'db_name')
        copyParameter(input_config, config, 'db_host')
        copyParameter(input_config, config, 'log_dir')
        copyParameter(input_config, config, 'sub_roi_offset')
        copyParameter(input_config, config, 'sub_roi_shape')

        if 'num_workers' in config:
            config['num_workers'] = int(config['num_workers'])

    if "PredictTask" in configs:
        config = configs["PredictTask"]
        config['raw_file'] = input_config['raw_file']
        config['raw_dataset'] = input_config['raw_dataset']
        if 'out_file' not in config:
            config['out_file'] = input_config['output_file']
        config['train_dir'] = network_config['train_dir']
        config['iteration'] = network_config['iteration']
        # config['log_dir'] = input_config['log_dir']
        # config['net_voxel_size'] = network_config['net_voxel_size']
        copyParameter(network_config, config, 'net_voxel_size')
        copyParameter(network_config, config, 'predict_file')
        # if 'predict_file' in network_config:
        #     config['predict_file'] = network_config['predict_file']
        # else:
        #     config['predict_file'] = "predict.py"
        if 'predict_file' not in config or config['predict_file'] is None:
            config['predict_file'] = "predict.py"
        if 'xy_downsample' in network_config:
            config['xy_downsample'] = network_config['xy_downsample']
        if 'roi_offset' in input_config:
            config['roi_offset'] = input_config['roi_offset']
        if 'roi_shape' in input_config:
            config['roi_shape'] = input_config['roi_shape']
        if 'roi_context' in input_config:
            config['roi_context'] = input_config['roi_context']
        config['myelin_prediction'] = network_config.get('myelin_prediction', 0)
        copyParameter(input_config, config, 'delete_section_list')
        copyParameter(input_config, config, 'replace_section_list')
        copyParameter(input_config, config, 'overwrite_sections')
        copyParameter(input_config, config, 'overwrite_mask_f')
        copyParameter(input_config, config, 'center_roi_offset')

    if "FixRawFromCatmaidTask" in configs:
        config = configs["FixRawFromCatmaidTask"]
        copyParameter(input_config, config, 'raw_file')
        copyParameter(input_config, config, 'raw_dataset')

    if "PredictMyelinTask" in configs:
        raise RuntimeError("Deprecated task")
        config = configs["PredictMyelinTask"]
        config['raw_file'] = input_config['raw_file']
        config['myelin_file'] = input_config['output_file']
        if 'roi_offset' in input_config:
            config['roi_offset'] = input_config['roi_offset']
        if 'roi_shape' in input_config:
            config['roi_shape'] = input_config['roi_shape']

    if "PredictCapillaryTask" in configs:
        config = configs["PredictCapillaryTask"]
        config['raw_file'] = input_config['raw_file']
        copyParameter(input_config, config, 'raw_dataset')
        config['out_file'] = input_config['output_file']
        if 'roi_offset' in input_config:
            config['roi_offset'] = input_config['roi_offset']
        if 'roi_shape' in input_config:
            config['roi_shape'] = input_config['roi_shape']
        copyParameter(input_config, config, 'replace_section_list')

    if "MergeMyelinTask" in configs:
        config = configs["MergeMyelinTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['output_file']
        config['myelin_file'] = input_config['output_file']
        config['merged_affs_file'] = input_config['output_file']
        config['log_dir'] = input_config['log_dir']

    if "DownsampleTask" in configs:
        config = configs["DownsampleTask"]
        copyParameter(input_config, config, 'output_file', 'affs_file')

    if "ExtractFragmentTask" in configs:
        config = configs["ExtractFragmentTask"]
        copyParameter(input_config, config, 'output_file', 'affs_file')
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        # copyParameter(input_config, config, 'output_file', 'capillary_pred_file')
        copyParameter(input_config, config, 'raw_file')
        copyParameter(input_config, config, 'raw_dataset')
        copyParameter(input_config, config, 'overwrite_sections')
        copyParameter(input_config, config, 'overwrite_mask_f')
        copyParameter(input_config, config, 'db_file_name')

    if "AgglomerateTask" in configs:
        config = configs["AgglomerateTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['output_file']
        config['fragments_file'] = input_config['output_file']
        config['merge_function'] = merge_function
        copyParameter(input_config, config, 'sub_roi_offset')
        copyParameter(input_config, config, 'sub_roi_shape')
        copyParameter(input_config, config, 'overwrite_sections')
        copyParameter(input_config, config, 'overwrite_mask_f')
        config['edges_collection'] = "edges_" + merge_function
        copyParameter(input_config, config, 'db_file_name')

    if "FindSegmentsBlockwiseTask" in configs:
        config = configs["FindSegmentsBlockwiseTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        config['edges_collection'] = "edges_" + merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut
        copyParameter(input_config, config, 'db_file_name')

    if "MakeInterThresholdMappingTask" in configs:
        config = configs["MakeInterThresholdMappingTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        config['edges_collection'] = "edges_" + merge_function

    if "FindSegmentsBlockwiseTask2" in configs:
        config = configs["FindSegmentsBlockwiseTask2"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut

    if "FindSegmentsBlockwiseTask2a" in configs:
        config = configs["FindSegmentsBlockwiseTask2a"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut

    if "FindSegmentsBlockwiseTask2b" in configs:
        config = configs["FindSegmentsBlockwiseTask2b"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut

    if "FindSegmentsBlockwiseTask3" in configs:
        config = configs["FindSegmentsBlockwiseTask3"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut

    if "FindSegmentsBlockwiseTask4" in configs:
        config = configs["FindSegmentsBlockwiseTask4"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut

    if "ExtractSegmentationFromLUT" in configs:
        config = configs["ExtractSegmentationFromLUT"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        copyParameter(input_config, config, 'output_file', 'out_file')

    if "ExtractSegmentationFromLUTBlockwiseTask" in configs:
        config = configs["ExtractSegmentationFromLUTBlockwiseTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        copyParameter(input_config, config, 'output_file', 'out_file')
        config['merge_function'] = merge_function

    if "ExtractSuperFragmentSegmentationTask" in configs:
        config = configs["ExtractSuperFragmentSegmentationTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        copyParameter(input_config, config, 'output_file', 'out_file')
        config['merge_function'] = merge_function

    network_config = configs["SynfulNetwork"]

    if "PredictSynapseTask" in configs:
        config = configs["PredictSynapseTask"]
        os.makedirs(input_config_synful['log_dir'], exist_ok=True)
        # print(input_config_synful); exit()
        config['raw_file'] = input_config_synful['raw_file']
        config['raw_dataset'] = input_config_synful['raw_dataset']
        if 'out_file' not in config:
            config['out_file'] = input_config_synful['output_file']
        copyParameter(network_config, config, 'train_dir')
        copyParameter(network_config, config, 'iteration')
        config['log_dir'] = input_config_synful['log_dir']
        copyParameter(network_config, config, 'net_voxel_size')
        config['predict_file'] = network_config.get(
            'predict_file', 'segway/synful_tasks/predict.py')
        copyParameter(network_config, config, 'predict_file')
        copyParameter(network_config, config, 'xy_downsample')
        copyParameter(input_config_synful, config, 'roi_offset')
        copyParameter(input_config_synful, config, 'roi_shape')
        copyParameter(input_config_synful, config, 'sub_roi_offset')
        copyParameter(input_config_synful, config, 'sub_roi_shape')
        copyParameter(input_config_synful, config, 'delete_section_list')
        copyParameter(input_config_synful, config, 'replace_section_list')
        copyParameter(input_config_synful, config, 'overwrite_sections')
        copyParameter(input_config_synful, config, 'overwrite_mask_f')
        copyParameter(input_config_synful, config, 'center_roi_offset')
        copyParameter(network_config, config, 'out_properties')

    if "ExtractSynapsesTask" in configs:
        config = configs["ExtractSynapsesTask"]
        # config['raw_file'] = input_config['raw_file']
        # config['raw_dataset'] = input_config['raw_dataset']
        copyParameter(input_config, config, 'sub_roi_offset')
        copyParameter(input_config, config, 'sub_roi_shape')
        copyParameter(input_config, config, 'output_file', 'super_fragments_file')
        copyParameter(input_config, config, 'output_file', 'syn_indicator_file')
        copyParameter(input_config, config, 'output_file', 'syn_dir_file')

    if "PredictSynapseDirTask" in configs:
        config = configs["PredictSynapseDirTask"]
        os.makedirs(input_config_synful1['log_dir'], exist_ok=True)
        config['raw_file'] = input_config_synful1['raw_file']
        config['raw_dataset'] = input_config_synful1['raw_dataset']
        if 'out_file' not in config:
            config['out_file'] = input_config_synful1['output_file']
        copyParameter(network_config, config, 'train_dir1', 'train_dir')
        copyParameter(network_config, config, 'iteration1', 'iteration')
        copyParameter(network_config, config, 'out_properties1', 'out_properties')
        config['log_dir'] = input_config_synful1['log_dir']
        copyParameter(network_config, config, 'net_voxel_size')
        config['predict_file'] = 'segway/synful_tasks/predict.py'
        copyParameter(network_config, config, 'predict_file')
        copyParameter(network_config, config, 'xy_downsample')
        copyParameter(input_config_synful1, config, 'roi_offset')
        copyParameter(input_config_synful1, config, 'roi_shape')
        copyParameter(input_config_synful1, config, 'sub_roi_offset')
        copyParameter(input_config_synful1, config, 'sub_roi_shape')
        copyParameter(input_config_synful1, config, 'delete_section_list')
        copyParameter(input_config_synful1, config, 'replace_section_list')
        copyParameter(input_config_synful1, config, 'overwrite_sections')
        copyParameter(input_config_synful1, config, 'overwrite_mask_f')
        copyParameter(input_config_synful1, config, 'center_roi_offset')


def compute_compatible_roi(
        roi_offset, roi_shape,
        sub_roi_offset, sub_roi_shape,
        roi_context,
        source_roi,
        chunk_size,
        center_roi_offset=False,
        shrink_context=True,
        sched_roi_outside_roi_ok=False,
        ):
    '''Compute compatible input (schedule) ROI and output (dataset ROI)'''

    roi_context = daisy.Coordinate(roi_context)

    if roi_offset is not None and roi_shape is not None:

        dataset_roi = daisy.Roi(
            tuple(roi_offset), tuple(roi_shape))

        if center_roi_offset:
            dataset_roi = dataset_roi.shift(-daisy.Coordinate(tuple(roi_shape))/2)
            dataset_roi = dataset_roi.snap_to_grid(voxel_size, mode="grow")

        sched_roi = dataset_roi.grow(roi_context, roi_context)
        # assert sched_roi.intersect(source_roi) == sched_roi, \
        #     "input_roi (%s) + roi_context (%s) = output_roi (%s) has to be within raw ROI %s" \
        #     % (dataset_roi, roi_context, sched_roi, source_roi)
        assert dataset_roi.intersect(source_roi) == dataset_roi, \
            "input_roi (%s) + roi_context (%s) = output_roi (%s) has to be within raw ROI %s" \
            % (dataset_roi, roi_context, dataset_roi, source_roi)

    elif sub_roi_offset is not None and sub_roi_shape is not None:

        dataset_roi = source_roi  # total volume ROI
        sched_roi = daisy.Roi(
            tuple(sub_roi_offset), tuple(sub_roi_shape))
        # assert dataset_roi.contains(sched_roi)

        if center_roi_offset:
            raise RuntimeError("Unimplemented")
        # need align dataset_roi to prediction chunk size

        output_roi_begin = [k for k in dataset_roi.get_begin()]
        output_roi_begin[0] = align(dataset_roi.get_begin()[0], sched_roi.get_begin()[0], chunk_size[0])
        output_roi_begin[1] = align(dataset_roi.get_begin()[1], sched_roi.get_begin()[1], chunk_size[1])
        output_roi_begin[2] = align(dataset_roi.get_begin()[2], sched_roi.get_begin()[2], chunk_size[2])

        print("dataset_roi:", dataset_roi)
        print("sched_roi:", sched_roi)
        print("chunk_size:", chunk_size)
        dataset_roi.set_offset(tuple(output_roi_begin))
        print("dataset_roi:", dataset_roi)

        assert (dataset_roi.get_begin()[0] - sched_roi.get_begin()[0]) % chunk_size[0] == 0
        assert (dataset_roi.get_begin()[1] - sched_roi.get_begin()[1]) % chunk_size[1] == 0
        assert (dataset_roi.get_begin()[2] - sched_roi.get_begin()[2]) % chunk_size[2] == 0

        if not sched_roi_outside_roi_ok:
            assert dataset_roi.contains(sched_roi), "dataset_roi %s does not contain sched_roi %s" % (dataset_roi, sched_roi)

        sched_roi = sched_roi.grow(roi_context, roi_context)

    else:

        if center_roi_offset:
            raise RuntimeError("Cannot center ROI if not specified")

        assert roi_offset is None
        assert roi_shape is None
        assert sub_roi_offset is None
        assert sub_roi_shape is None
        # if no ROI is given, we need to shrink output ROI
        # to account for the roi_context
        sched_roi = source_roi
        dataset_roi = source_roi

        if shrink_context:
            dataset_roi = dataset_roi.grow(-roi_context, -roi_context)

    return sched_roi, dataset_roi


def align(a, b, stride):
    # align a to b such that b - a is multiples of stride
    assert b >= a
    print(a)
    print(b)
    l = b - a
    print(l)
    l = int(l/stride) * stride
    print(l)
    print(b - l)
    return b - l


def check_block(
        block,
        vol_ds,
        is_precheck,
        completion_db,
        recording_block_done,
        logger,
        overwrite_sections=None,
        overwrite_mask=None,
        check_datastore=True,
        ):

    logger.debug("Checking if block %s is complete..." % block.write_roi)

    write_roi = vol_ds.roi.intersect(block.write_roi)
    if write_roi.empty():
        logger.debug("Block outside of output ROI")
        return True

    if is_precheck and overwrite_sections is not None:
        # read_roi_mask = overwrite_mask.roi.intersect(block.read_roi)
        for roi in overwrite_sections:
            # if roi.intersects(block.read_roi):
            if roi.intersects(block.write_roi):
                logger.debug("Block overlaps overwrite_sections %s" % roi)
                return False

    if is_precheck and overwrite_mask:
        read_roi_mask = overwrite_mask.roi.intersect(block.read_roi)
        if not read_roi_mask.empty():
            try:
                sum = np.sum(overwrite_mask[read_roi_mask].to_ndarray())
                if sum != 0:
                    logger.debug("Block inside overwrite_mask")
                    return False
            except:
                return False

    if completion_db.count({'block_id': block.block_id}) >= 1:
        logger.debug("Skipping block with db check")
        return True

    if check_datastore:
        s = 0
        quarter = (write_roi.get_end() - write_roi.get_begin()) / 4

        # check values of center and nearby voxels
        s += np.sum(vol_ds[write_roi.get_begin() + quarter*1])
        s += np.sum(vol_ds[write_roi.get_begin() + quarter*2])
        s += np.sum(vol_ds[write_roi.get_begin() + quarter*3])
        logger.info("Sum of center values in %s is %f" % (write_roi, s))

        done = s != 0
        if done:
            recording_block_done(block)
        return done
    else:
        return False
