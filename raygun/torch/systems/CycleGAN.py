#%%
from copy import deepcopy
import itertools
import logging
import random
from matplotlib import pyplot as plt
import torch
import glob
import zarr
import daisy
import os

import gunpowder as gp
import math
import functools
from tqdm import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True

from raygun.torch.models import CycleModel
from raygun.torch.losses import LinkCycleLoss, SplitCycleLoss
from raygun.torch.optimizers import BaseDummyOptimizer
from raygun.torch.systems import BaseSystem

class CycleGAN(BaseSystem):
    def __init__(self, config=None):
        super().__init__(default_config='../default_configs/default_cycleGAN_conf.json', config=config)
        self.logger = logging.Logger(__name__, 'INFO')

        if self.common_voxel_size is None:
            self.common_voxel_size = gp.Coordinate(daisy.open_ds(self.sources['B'].path, self.sources['B'].name).voxel_size)
        else:
            self.common_voxel_size = gp.Coordinate(self.common_voxel_size)
        if self.ndims is None:
            self.ndims = sum(np.array(self.common_voxel_size) == np.min(self.common_voxel_size))
            
        self.build_machine()
        self.training_pipeline = None
        self.test_training_pipeline = None    

    def batch_show(self, batch=None, i=0, show_mask=False):
        if batch is None:
            batch = self.batch
        if not hasattr(self, 'col_dict'): 
            self.col_dict = {'REAL':0, 'FAKE':1, 'CYCL':2}
        if show_mask: self.col_dict['MASK'] = 3
        rows = (self.real_A in batch.arrays) + (self.real_B in batch.arrays)       
        cols = 0
        for key in self.col_dict.keys():
            cols += key in [array.identifier[:4] for array in batch.arrays] 
        fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 10*rows))
        for array, value in batch.items():
            label = array.identifier
            if label[:4] in self.col_dict:
                c = self.col_dict[label[:4]]
                r = (int('_B' in label) + int('FAKE' in label)) % 2
                if len(value.data.shape) > 3: # pick one from the batch
                    img = value.data[i].squeeze()
                else:
                    img = value.data.squeeze()
                if len(img.shape) == 3:
                    mid = img.shape[0] // 2 # for 3D volume
                    data = img[mid]
                else:
                    data = img
                if rows == 1:
                    axes[c].imshow(data, cmap='gray', vmin=0, vmax=1)
                    axes[c].set_title(label)                
                else:
                    axes[r, c].imshow(data, cmap='gray', vmin=0, vmax=1)
                    axes[r, c].set_title(label)

    def write_tBoard_graph(self, batch=None):
        if batch is None:
            batch = self.batch
        
        ex_inputs = []
        if self.real_A in batch:
            ex_inputs += [torch.tensor(batch[self.real_A].data)]
        if self.real_B in batch:
            ex_inputs += [torch.tensor(batch[self.real_B].data)]

        for i, ex_input in enumerate(ex_inputs):
            if self.ndims == len(self.common_voxel_size): # add channel dimension if necessary
                ex_input = ex_input.unsqueeze(axis=1)
            if self.batch_size == 1: # ensure batch dimension is present
                ex_input = ex_input.unsqueeze(axis=0)
            ex_inputs[i] = ex_input
        
        try:
            self.trainer.summary_writer.add_graph(self.model, ex_inputs)                
        except:
            self.logger.warning('Failed to add model graph to tensorboard.')

    def get_extents(self, side_length=None, array_name=None):
        if side_length is None:
            side_length = self.side_length

        if ('padding_type' in self.gnet_kwargs) and (self.gnet_kwargs['padding_type'].lower() == 'valid'):
            if array_name is not None and not ('real' in array_name.lower() or 'mask' in array_name.lower()):
                shape = (1,1) + (side_length,) * self.ndims
                pars = [par for par in self.netG1.parameters()]
                result = self.netG1(torch.zeros(*shape, device=pars[0].device))
                if 'fake' in array_name.lower():
                    side_length = result.shape[-1]
                elif 'cycle' in array_name.lower():
                    result = self.netG1(result)
                    side_length = result.shape[-1]

        extents = np.ones((len(self.common_voxel_size)))
        extents[-self.ndims:] = side_length # assumes first dimension is z (i.e. the dimension breaking isotropy)
        return gp.Coordinate(extents)

    def setup_networks(self):
        self.netG1 = self.get_network(self.gnet_type, self.gnet_kwargs)
        self.netG2 = self.get_network(self.gnet_type, self.gnet_kwargs)
        
        self.netD1 = self.get_network(self.dnet_type, self.dnet_kwargs)
        self.netD2 = self.get_network(self.dnet_type, self.dnet_kwargs)
        
    def setup_model(self):
        if not hasattr(self, 'netG1'):
            self.setup_networks()

        if self.sampling_bottleneck:
            scale_factor_A = tuple(np.divide(self.common_voxel_size, self.A_voxel_size)[-self.ndims:])
            if not any([s < 1 for s in scale_factor_A]): scale_factor_A = None
            scale_factor_B = tuple(np.divide(self.common_voxel_size, self.B_voxel_size)[-self.ndims:])
            if not any([s < 1 for s in scale_factor_B]): scale_factor_B = None
        else:
            scale_factor_A, scale_factor_B = None, None
        
        self.model = CycleModel(self.netG1, self.netG2, scale_factor_A, scale_factor_B, split=self.loss_type.lower()=='split')
    
    def setup_optimization(self):
        if isinstance(self.optim, str):
            base_optim = getattr(torch.optim, self.optim)
        else:
            base_optim = self.optim

        self.optimizer_D = base_optim(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), **self.d_optim_kwargs)

        if self.loss_type.lower()=='link':                        
            self.optimizer_G = base_optim(itertools.chain(self.netG1.parameters(), self.netG2.parameters()), **self.g_optim_kwargs)
            self.optimizer = BaseDummyOptimizer(optimizer_G=self.optimizer_G, optimizer_D=self.optimizer_D) #TODO: May be unecessary to pass actual optimizers
            
            self.loss = LinkCycleLoss(self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_G, self.optimizer_D, self.ndims, **self.loss_kwargs)        
    
        elif self.loss_type.lower()=='split':                 
            self.optimizer_G1 = base_optim(self.netG1.parameters(), **self.g_optim_kwargs)
            self.optimizer_G2 = base_optim(self.netG2.parameters(), **self.g_optim_kwargs)
            self.optimizer = BaseDummyOptimizer(optimizer_G1=self.optimizer_G1, optimizer_G2=self.optimizer_G2, optimizer_D=self.optimizer_D)
            
            self.loss = SplitCycleLoss(self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_G1, self.optimizer_G2, self.optimizer_D, self.ndims, **self.loss_kwargs)        
    
        else:
            raise NotImplementedError("Unexpected Loss Style. Accepted options are 'cycle' or 'split'")

    def build_machine(self):       
        # initialize needed variables
        self.arrays = []

        # define our network model for training
        self.setup_networks()        
        self.setup_model()

        # get performance stats
        self.performance = gp.PrintProfilingStats(every=self.log_every)

        # setup a cache
        self.cache = gp.PreCache(num_workers=self.num_workers, cache_size=self.cache_size)#os.cpu_count())

        # define axes for mirroring and transpositions
        self.augment_axes = list(np.arange(3)[-self.ndims:]) #TODO: Maybe limit to xy?

        # build datapipes
        self.datapipe_A = self.get_datapipe('A')
        self.datapipe_B = self.get_datapipe('B') #datapipe has: train_pipe, source, reject, resample, augment, unsqueeze, etc.}

    def build_training_pipeline(self):
        # create a train node using our model, loss, and optimizer
        self.trainer = gp.torch.Train(
                            self.model,
                            self.loss,
                            self.optimizer,
                            inputs = {
                                'real_A': self.real_A,
                                'real_B': self.real_B
                            },
                            outputs = {
                                0: self.fake_B,
                                1: self.cycled_B,
                                2: self.fake_A,
                                3: self.cycled_A
                            },
                            loss_inputs = {
                                0: self.real_A,
                                1: self.fake_A,
                                2: self.cycled_A,
                                3: self.real_B,
                                4: self.fake_B,
                                5: self.cycled_B,
                            },
                            log_dir=self.tensorboard_path,
                            log_every=self.log_every,
                            checkpoint_basename=self.model_path+self.model_name,
                            save_every=self.save_every,
                            spawn_subprocess=self.spawn_subprocess
                            )

        # assemble pipeline
        self.training_pipeline = (self.pipe_A, self.pipe_B) + gp.MergeProvider() #merge upstream pipelines for two sources
        self.training_pipeline += self.trainer        
        self.training_pipeline += self.datapipe_A.postnet_pipe.cycle + self.datapipe_B.postnet_pipe.cycle
        if self.batch_size == 1:
            self.training_pipeline += gp.Squeeze([self.real_A, 
                                            self.fake_A, 
                                            self.cycled_A, 
                                            self.real_B, 
                                            self.fake_B, 
                                            self.cycled_B
                                            ], axis=0)
        self.test_training_pipeline = self.training_pipeline.copy() + self.performance
        self.training_pipeline += self.cache

        # create request
        self.train_request = gp.BatchRequest()
        for array in self.arrays:            
            extents = self.get_extents(array_name=array.identifier)
            self.train_request.add(array, self.common_voxel_size * extents, self.common_voxel_size)
            
    def test(self, mode='train'):
        if self.test_training_pipeline is None:
            self.build_training_pipeline()
        getattr(self.model, mode)() # set to 'train' or 'eval'
        with gp.build(self.test_training_pipeline):
            self.batch = self.test_training_pipeline.request_batch(self.train_request)
        self.batch_show()
        return self.batch

    def train(self):
        if self.training_pipeline is None:
            self.build_training_pipeline()
        self.model.train()
        with gp.build(self.training_pipeline):
            for i in tqdm(range(self.num_epochs)):
                # this_request = copy.deepcopy(self.train_request)
                # this_request._random_seed = random.randint(0, 2**32)
                # self.batch = self.training_pipeline.request_batch(this_request)
                self.batch = self.training_pipeline.request_batch(self.train_request)
                if i == 1:
                    self.write_tBoard_graph()
                if hasattr(self.loss, 'loss_dict'):
                    print(self.loss.loss_dict)
                if i % self.log_every == 0:
                    self.batch_tBoard_write()
        return self.batch


# %%
