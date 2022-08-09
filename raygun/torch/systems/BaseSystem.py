import functools
from glob import glob
import re
import logging
import os
from random import random

import numpy as np
import torch

from raygun.torch.utils import read_config
from raygun.torch import networks
from raygun.torch.networks.utils import init_weights

class BaseSystem:
    def __init__(self, default_config=f'{os.path.dirname(os.path.dirname(__file__))}/default_configs/blank_conf.json', config=None):
        #Add default params
        for key, value in read_config(os.path.abspath(default_config)).items():
            setattr(self, key, value)
        
        if config is not None:
            #Get this configuration
            for key, value in read_config(os.path.abspath(config)).items():
                setattr(self, key, value)
                
        if self.checkpoint is None:
            try:
                self.checkpoint, self.iteration = self._get_latest_checkpoint()
            except:
                print('Checkpoint not found. Starting from scratch.')
                self.checkpoint = None

        if self.random_seed is not None:
            self.set_random_seed()

    def set_random_seed(self):
        if self.random_seed is None:
            self.random_seed = 42
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)    
    
    def set_verbose(self, verbose=None):
        if verbose is not None:
            self.verbose = verbose
        elif self.verbose is None:
            self.verbose = True
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

    def _get_latest_checkpoint(self):
        basename = self.model_path + self.model_name
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        checkpoints = glob(basename + '_checkpoint_*')
        checkpoints.sort(key=natural_keys)

        if len(checkpoints) > 0:

            checkpoint = checkpoints[-1]
            iteration = int(checkpoint.split('_')[-1])
            return checkpoint, iteration

        return None, 0
    
    def batch_tBoard_write(self):
        self.trainer.summary_writer.flush()
        self.n_iter = self.trainer.iteration

    def get_downsample_factors(self, net_kwargs):
        if 'downsample_factors' not in net_kwargs:
            down_factor = 2 if 'down_factor' not in net_kwargs else net_kwargs.pop('down_factor')
            num_downs = 3 if 'num_downs' not in net_kwargs else net_kwargs.pop('num_downs')
            net_kwargs.update({'downsample_factors': [(down_factor,)*self.ndims,] * (num_downs - 1)})
        return net_kwargs

    def get_network(self, net_type='unet', net_kwargs=None):
        if net_type == 'unet':
            net_kwargs = self.get_downsample_factors(net_kwargs)

            net = torch.nn.Sequential(
                                networks.UNet(**net_kwargs),
                                torch.nn.Tanh()
                                )        
        elif net_type == 'residualunet':
            net_kwargs = self.get_downsample_factors(net_kwargs)
            
            net = torch.nn.Sequential(
                                networks.ResidualUNet(**net_kwargs), 
                                torch.nn.Tanh()
                                )            
        elif net_type == 'resnet':
            net = networks.ResNet(**net_kwargs)
        elif net_type == 'classic':
            norm_instance = {
                                2: torch.nn.InstanceNorm2d,
                                3: torch.nn.InstanceNorm3d,
                            }[self.ndims]
            net_kwargs['norm_layer'] = functools.partial(norm_instance, affine=False, track_running_stats=False)
            net = networks.NLayerDiscriminator(**net_kwargs)
        elif hasattr(networks, net_type):
            net = getattr(networks, net_type)(**net_kwargs)
        else:
            raise f'Unknown discriminator type requested: {net_type}'
        
        activation = net_kwargs['activation'] if 'activation' in net_kwargs else torch.nn.ReLU        
        if activation is not None:
            init_weights(net, init_type='kaiming', nonlinearity=activation.__class__.__name__.lower())
        elif net_type == 'classic':
            init_weights(net, init_type='kaiming')
        else:
            init_weights(net, init_type='normal', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE

        return net
        
    def get_valid_context(self, net_kwargs, side_length=None):
        # returns number of pixels to crop from a side to trim network outputs to valid FOV
        if side_length is None:
            side_length = self.side_length
        
        net_kwargs['padding_type'] = 'valid'
        net = self.get_network(gnet_kwargs=net_kwargs)
        
        shape = (1,1) + (side_length,) * self.ndims
        pars = [par for par in net.parameters()]
        result = net(torch.zeros(*shape, device=pars[0].device))
        return np.ceil((np.array(shape) - np.array(result.shape)) / 2)[-self.ndims:]

    def setup_networks(self):
        '''Implement network setups in subclasses.'''
        raise NotImplementedError()
        
    def setup_model(self):
        '''Implement model setup in subclasses.'''
        raise NotImplementedError()
        
    def setup_optimization(self):
        '''Implement optimization setup in subclasses.'''
        raise NotImplementedError()
    
    def build_system(self):
        # initialize needed variables
        self.arrays = []

        # define our network model for training
        self.setup_networks()        
        self.setup_model()
        self.setup_optimization()
        ...#TODO