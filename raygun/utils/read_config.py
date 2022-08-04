#%%
from functools import partial
from io import StringIO
import json
import os
import torch

from jsmin import jsmin

def loadJsonFile(fin):
    with open(fin, 'r') as f:
        config = json.load(StringIO(jsmin(f.read())))
    return config

def _eval_args(arg):
    parts = []
    inds = []
    while arg.count('#') > 0:
        if len(inds) == 0:
            inds.append(arg.find('#'))
        elif len(inds) == 1:
            inds.append(arg.find('#', inds[0]+1))
        if len(inds) == 2:
            parts.append(arg[:inds[0]])
            parts.append(eval(arg[inds[0]+1:inds[-1]]))            
            arg = arg[inds[-1]+1:]
            inds = []
    return ''.join(parts)

def eval_args(config, file):
    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = eval_args(v, file)
        elif isinstance(v, str):
            if '$working_dir' in v:
                v = v.replace('$working_dir', os.path.dirname(file))
            if v[0] == '#' and v[-1] == '#':
                v = eval(v[1:-1])
            elif v.count('#') > 0 and v.count('#') % 2 == 0:
                v = _eval_args(v)            
            config[k] = v
    return config

def read_config(file):
    configs = []
    configs.append(loadJsonFile(file))
    last_file = file
    while 'include' in configs[-1].keys():
        include_file = configs[-1]['include']
        if '..' in include_file:
            include_file = include_file.replace('..', os.path.dirname(os.path.dirname(last_file)))
        configs.append(loadJsonFile(include_file))
        last_file = include_file
    
    config = {}
    for c in configs[-1::-1]:
        config.update(**c)
    
    config = eval_args(config, file)
    
    return config

# %%
