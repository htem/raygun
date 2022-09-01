#!/usr/bin/env python3
#SBATCH -t 04:00:00
#SBATCH -p short 
#SBATCH -o igneous_logs/downsample.o%j
#SBATCH -e igneous_logs/downsample.e%j
#SBATCH -J downsample_CloudVolume
#SBATCH --mail-type=END
#SBATCH --mem 32GB

# First generate tasks by running:
#   ./downsample.py create_task_queue
# You can submit this as a job to the o2 cluster by running:
#   sbatch downsample.py run_tasks_from_queue
# Run that command multiple times (4 to 16 times is reasonable) to process tasks in parallel

import sys

import igneous.task_creation as tc
from taskqueue import TaskQueue, LocalTaskQueue


#cloud_bucket = 'gs://lee-pacureanu_data-exchange_us-storage'
#cloud_folder = 'ls2892_LTP/2102/ctx_eosin/ctx_eosin_tile02_070nm_rec_db9_.raw'
#cloud_path = cloud_bucket + '/' + cloud_folder
#queuepath = volume_fn.split('/')[-1].split('.')[0]+'igneous_tasks'

def create_task_queue(cloud_path=None,
                    mip=0,       # Starting mip
                    num_mips=4,  # Final mip to downsample to
                    bounds=None, # None will use full bounds
                    factor=(2, 2, 2),  # Downsample all 3 axes
                    queuepath = 'igneous_tasks'
                ):
    tq = TaskQueue('fq://'+queuepath)
    if cloud_path is None:
        cloud_path=input('Cloud Path:')
    tasks = tc.create_downsampling_tasks(
        cloud_path,
        mip=mip,       # Starting mip
        num_mips=num_mips,  # Final mip to downsample to
        bounds=bounds,
        factor=factor  # Downsample all 3 axes
    )
    tq.insert(tasks)
    print('Done adding {} tasks to queue at {}'.format(len(tasks), queuepath))


def run_tasks_from_queue(queuepath = 'igneous_tasks'):
    tq = TaskQueue('fq://'+queuepath)
    print('Working on tasks from filequeue "{}"'.format(queuepath))
    tq.poll(
        verbose=True, # prints progress
        lease_seconds=3000,
        tally=True # makes tq.completed work, logs 1 byte per completed task
    )
    print('Done')


def run_tasks_locally(cloud_path=None,
                    mip=0,       # Starting mip
                    num_mips=4,  # Final mip to downsample to
                    bounds=None, # None will use full bounds
                    factor=(2, 2, 2),  # Downsample all 3 axes
                    n_cores=4):
    tq = LocalTaskQueue(parallel=n_cores)
    if cloud_path is None:
        cloud_path=input('Cloud Path:')
    tasks = tc.create_downsampling_tasks(
        cloud_path,
        mip=mip,       # Starting mip
        num_mips=num_mips,  # Final mip to downsample to
        bounds=bounds,
        factor=factor  # Downsample all 3 axes
    )
    tq.insert(tasks)
    print('Running in-memory task queue on {} cores'.format(n_cores))
    tq.execute()
    print('Done')


if __name__ == '__main__':
    l = locals()
    public_functions = [f for f in l if callable(l[f]) and f[0] != '_']
    if len(sys.argv) <= 1 or not sys.argv[1] in public_functions:
        from inspect import signature
        print('Functions available:')
        for f_name in public_functions:
            print('  '+f_name+str(signature(l[f_name])))
            docstring = l[f_name].__doc__
            if not isinstance(docstring, type(None)):
                print(docstring.strip('\n'))
        # TODO add an instruction here that says:
        # 'For example, run the following from your command line to call the function blah:'
        # 'python script_name.py blah arg1 arg2 kw1=kwarg1'
    else:
        func = l[sys.argv[1]]
        args = []
        kwargs = {}
        for arg in sys.argv[2:]:
            if '=' in arg:
                split = arg.split('=')
                kwargs[split[0]] = split[1]
            else:
                args.append(arg)
        func(*args, **kwargs)