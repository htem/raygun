#%% Imports and settings
import os
import tempfile
import numpy as np
from raygun.torch.systems import CycleGAN

# os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

config_path = '/n/groups/htem/users/jlr54/raygun/experiments/ieee-isbi-2022/01_cycle_gans/test_conf.json' #TODO: Use importlib.resources instead
system = CycleGAN(config_path)

#%% Setup
cur_dir = os.getcwd()
temp_dir = tempfile.TemporaryDirectory()
os.chdir(temp_dir.name)
print(f'Executing test in {os.getcwd()}')

#%% Test prenet_pipe
system.build_system()
pnet = system.trainer.prenet_pipe('test')
req = system.make_request('prenet')
import gunpowder as gp
with gp.build(pnet):
    batch = pnet.request_batch(req)

system.batch_show(batch)
system.arrays_min_max(batch, lims={np.float32:[-1,1]}, show=True)

#%% Test single batch
batch = system.test()

system.arrays_min_max(show=True)
system.trainer.print_profiling_stats()

#%% Full test
system.train()

print(f'View tensorboard by running the following:\n\ttensorboard --logdir {os.getcwd()}/tensorboard &')

# %% Cleanup
os.chdir(cur_dir)
temp_dir.cleanup()
