import pymongo
import json
import hashlib
import multiprocessing
import os
import argparse
import copy

import daisy


class LaunchableDaisyTask():

    def parse_args(self, ap):

        try:
            ap.add_argument(
                "--db_host", type=str, help='database host',
                default='10.117.28.139')
                # default='10.117.28.250')
            ap.add_argument(
                "--db_name", type=str, help='database project name',
                default=None)
            ap.add_argument(
                "--completion_db_col", type=str, help='completion_db_col name',
                default=None)
            ap.add_argument(
                "--overwrite", type=int, help='completion_db_col name',
                default=0)
            ap.add_argument(
                "--num_workers", type=int, help='completion_db_col name',
                default=1)
            ap.add_argument(
                "--no_launch_workers", type=int, help='completion_db_col name',
                default=0)
            ap.add_argument(
                "--config_hash", type=str, help='config string, used to keep track of progress',
                default=None)
        except argparse.ArgumentError as e:
            print("Conflicting argument naming with LaunchableDaisyTask")
            raise e

        return vars(ap.parse_args())

    def init(self, config):
        self.launch_process_cmd = multiprocessing.Manager().dict()

        for key in config:
            setattr(self, '%s' % key, config[key])

        self.__init_config = copy.deepcopy(config)

        self._init(config)

    def _init(self, config):
        assert False, "Fn needs to be implemented by subclass"
    def worker_function(self, args):
        assert False, "Fn needs to be implemented by subclass"
    def schedule_blockwise(self):
        assert False, "Fn needs to be implemented by subclass"

    def run_worker(self, config_file):

        assert 'DAISY_CONTEXT' in os.environ, "DAISY_CONTEXT must be defined as an environment variable"
        print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])

        print("config_file:", config_file)
        with open(config_file, 'r') as f:
            run_config = json.load(f)

        print("run_config:", run_config)
        for key in run_config:
            # globals()['%s' % key] = run_config[key]
            setattr(self, '%s' % key, run_config[key])

        client_scheduler = daisy.Client()
        db_client = pymongo.MongoClient(self.db_host)
        db = db_client[self.db_name]
        completion_db = db[self.completion_db_col]

        self._init(run_config)

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            # logger.debug("Processing", block)

            self.worker_function(block)

            if completion_db is not None:
                document = {
                    'block_id': block.block_id
                }
                completion_db.insert(document)

            client_scheduler.release_block(block, ret=0)

    def write_config(self, extra_config=None):

        config = self.__init_config
        if extra_config:
            for k in extra_config:
                config[k] = extra_config[k]

        config_str = ''.join(['%s' % (v,) for v in config.values()])
        if self.config_hash is None:
            self.config_hash = str(abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16)))
        task_id = '%s_%s' % (self.__class__.__name__, self.config_hash)
        self.config_file = os.path.join(
            '.run_configs', '%s.config' % task_id)

        self.new_actor_cmd = 'python %s run_worker %s' % (self.worker_script_file, self.config_file)

        if self.db_name is None:
            self.db_name = '%s' % task_id
        if self.completion_db_col is None:
            self.completion_db_col = "completion_db_col"

        config['db_name'] = self.db_name
        config['completion_db_col'] = self.completion_db_col
        config['db_host'] = self.db_host

        try:
            os.makedirs('.run_configs')
        except Exception:
            pass
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

        self.sbatch_file = os.path.join('.run_configs', '%s.sh' % task_id)
        self.generateSbatchScript(
            self.sbatch_file,
            self.new_actor_cmd,
            log_dir='.logs',
            logname=task_id,
            )

    def generateSbatchScript(
            self,
            output_script,
            run_cmd,
            log_dir,
            logname,
            cpu_time=11,
            queue='short',
            cpu_cores=1,
            cpu_mem=2,
            gpu=None):
        text = []
        text.append("#!/bin/bash")
        text.append("#SBATCH -t %d:40:00" % cpu_time)

        if gpu is not None:
            text.append("#SBATCH -p gpu")
            if gpu == '' or gpu == 'any':
                text.append("#SBATCH --gres=gpu:1")
            else:
                text.append("#SBATCH --gres=gpu:{}:1".format(gpu))
        else:
            text.append("#SBATCH -p %s" % queue)
        text.append("#SBATCH -c %d" % cpu_cores)
        text.append("#SBATCH --mem=%dGB" % cpu_mem)
        text.append("#SBATCH -o {}/{}_%j.out".format(log_dir, logname))
        text.append("#SBATCH -e {}/{}_%j.err".format(log_dir, logname))

        text.append("")
        text.append(run_cmd)

        with open(output_script, 'w') as f:
            f.write('\n'.join(text))

    def _run_daisy(
            self,
            total_roi,
            read_roi,
            write_roi,
            check_fn=None,
            fit='shrink',
            ):

        print("Processing total_roi %s with read_roi %s and write_roi %s" % (total_roi, read_roi, write_roi))

        if check_fn is None:
            precheck = lambda b: self._default_check_fn(b)
            postcheck = lambda b: self._default_check_fn(b)
            check_fn = (precheck, postcheck)

        db_client = pymongo.MongoClient(self.db_host)

        if self.overwrite:
            print("Dropping %s in %s" % (self.db_name, self.db_host))

            if self.overwrite == 2:
                i = "Yes"
            else:
                i = input("Sure? Yes/[No] ")

            if i == "Yes":
                db_client.drop_database(self.db_name)
                print("Dropped %s!" % self.db_name)
            else:
                print("Aborted")
                exit(0)

        db = db_client[self.db_name]

        if self.completion_db_col not in db.list_collection_names():
            self.completion_db = db[self.completion_db_col]
            self.completion_db.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
        else:
            self.completion_db = db[self.completion_db_col]

        # if self.overwrite:
        #     check_fn = None

        # if completion_db is not None:
        #     precheck = lambda b: check_block(b, out_array, completion_db)

        daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=self._new_worker,
            check_function=check_fn,
            read_write_conflict=False,
            num_workers=self.num_workers,
            max_retries=1,
            fit=fit,
            periodic_callback=self._periodic_callback)

    def _default_check_fn(self, block):

        write_roi = self.ds.roi.intersect(block.write_roi)
        if write_roi.empty():
            return True
        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            return True
        return False

    def _new_worker(self):

        context_str = os.environ['DAISY_CONTEXT']
        if 'context_str' not in self.launch_process_cmd:
            self.launch_process_cmd['context_str'] = context_str

        self._periodic_callback()

        if not self.no_launch_workers:
            self.run_worker(self.config_file)

    def _periodic_callback(self):

        if "context_str" in self.launch_process_cmd:
            print("sbatch command: DAISY_CONTEXT={} sbatch --parsable {}".format(
                self.launch_process_cmd["context_str"], self.sbatch_file))
            print("Local run command: DAISY_CONTEXT={} {}".format(
                self.launch_process_cmd["context_str"], self.new_actor_cmd))
