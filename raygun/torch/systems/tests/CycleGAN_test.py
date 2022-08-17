#%%
import os
import tempfile
from raygun.torch.systems import CycleGAN
# import torch

# torch.cuda.set_device(3)
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

config_path = '/n/groups/htem/users/jlr54/raygun/experiments/ieee-isbi-2022/01_cycle_gans/test_conf.json' #TODO: make relative path
system = CycleGAN(config_path)
#%%
cur_dir = os.getcwd()
temp_dir = tempfile.TemporaryDirectory()
os.chdir(temp_dir.name)

print(f'Executing test in {os.getcwd()}')

#%%
batch = system.test()

#%%
system.train()

# %%
os.chdir(cur_dir)
temp_dir.cleanup()