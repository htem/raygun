import logging
import os
from random import random

import numpy as np
import torch

from raygun.torch.utils import read_config

class BaseSystem:
    def __init__(self, default_config=f'{os.path.dirname(os.path.dirname(__file__))}/default_configs/blank_conf.json', config=None):
        #Add default params
        for key, value in read_config(os.path.abspath(default_config)).items():
            setattr(self, key, value)
        
        if config is not None:
            #Get this configuration
            for key, value in read_config(os.path.abspath(config)).items():
                setattr(self, key, value)

        self.logger = logging.Logger(__name__, 'INFO')        

    def set_random_seed(self):
        if self.random_seed is None:
            self.random_seed = 42
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)    
    
    def set_verbose(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def set_device(self, id=0):
        self.device_id = id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
        torch.cuda.set_device(id)

    def load_saved_model(self, checkpoint=None, cuda_available=None):
        if cuda_available is None:
            cuda_available = torch.cuda.is_available()
        if checkpoint is None:
            checkpoint = self.checkpoint
        else:
            self.checkpoint = checkpoint

        if checkpoint is not None:
            if not cuda_available:
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.logger.warning('No saved checkpoint found.')

