import itertools
import logging
from matplotlib import pyplot as plt
from raygun.io import CycleDataPipe
import torch
import daisy

import gunpowder as gp
import numpy as np

torch.backends.cudnn.benchmark = True

from raygun.torch.models import NotACycleModel
from raygun.torch.losses import NotACycleLoss
from raygun.torch.optimizers import BaseDummyOptimizer, get_base_optimizer
from raygun.torch.systems import BaseSystem

class NotACycleGAN(BaseSystem):
    def __init__(self, config=None):        
        super().__init__(default_config='../default_configs/default_NaC_conf.json', config=config)
        self.logger = logging.Logger(__name__, 'INFO')

        if self.common_voxel_size is None:
            self.common_voxel_size = gp.Coordinate(daisy.open_ds(self.sources['B']['path'], self.sources['B']['name']).voxel_size)
        else:
            self.common_voxel_size = gp.Coordinate(self.common_voxel_size)
        if self.ndims is None:
            self.ndims = sum(np.array(self.common_voxel_size) == np.min(self.common_voxel_size))
        
    def batch_show(self, batch=None, i=0, show_mask=False):
        if batch is None:
            batch = self.batch
        if not hasattr(self, 'col_dict'): 
            self.col_dict = {'REAL':0, 'FAKE':1, 'CYCL':2}
        if show_mask: self.col_dict['MASK'] = 3
        rows = (self.arrays['real_A'] in batch.arrays) + (self.arrays['real_B'] in batch.arrays)       
        cols = 0
        for key in self.col_dict.keys():
            cols += key in [array.identifier[:4] for array in batch.arrays] 
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
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
                    ax = axes[c]
                elif cols == 1:
                    ax = axes[r]      
                else:
                    ax = axes[r, c]
                
                ax.imshow(data, cmap='gray', vmin=-int(data.min()<0), vmax=1+254*int(data.max()>1))
                ax.set_title(label)

    def write_tBoard_graph(self, batch=None):
        if batch is None:
            batch = self.trainer.batch
        
        ex_inputs = []
        for id in ['A', 'B']:
            array = self.datapipes[id].real
            if array in batch:
                ex_inputs += [torch.tensor(batch[array].data)]

        for i, ex_input in enumerate(ex_inputs):
            if self.ndims == len(self.common_voxel_size): # add channel dimension if necessary
                ex_input = ex_input.unsqueeze(axis=1)
            if self.batch_size == 1: # ensure batch dimension is present
                ex_input = ex_input.unsqueeze(axis=0)
            ex_inputs[i] = ex_input
        
        try:
            self.trainer.train_node.summary_writer.add_graph(self.model, ex_inputs)                
        except:
            self.logger.warning('Failed to add model graph to tensorboard.')

    def get_extents(self, side_length=None, array_name=None):
        if side_length is None:
            side_length = self.side_length

        #TODO: Make work for 'VALID' padding
        # if ('padding_type' in self.gnet_kwargs) and (self.gnet_kwargs['padding_type'].lower() == 'valid'):
        #     if array_name is not None and not ('real' in array_name.lower() or 'mask' in array_name.lower()):
        #         shape = (1,1) + (side_length,) * self.ndims
        #         pars = [par for par in self.netG1.parameters()]
        #         result = self.netG1(torch.zeros(*shape, device=pars[0].device))
        #         if 'fake' in array_name.lower():
        #             side_length = result.shape[-1]
        #         elif 'cycle' in array_name.lower():
        #             result = self.netG1(result)
        #             side_length = result.shape[-1]

        extents = np.ones((len(self.common_voxel_size)))
        extents[-self.ndims:] = side_length # assumes first dimension is z (i.e. the dimension breaking isotropy)
        return gp.Coordinate(extents)

    def setup_networks(self):
        self.netD = self.get_network(self.dnet_type, self.dnet_kwargs)
        
    def setup_model(self):
        if not hasattr(self, 'netD'):
            self.setup_networks()
        
        self.model = NotACycleModel()
    
    def setup_optimization(self):
        self.optimizer_D = get_base_optimizer(self.d_optim_type)(self.netD.parameters(), **self.d_optim_kwargs)

        if hasattr(self, 'scheduler'):
            scheduler = self.scheduler
            scheduler_kwargs = self.scheduler_kwargs
        else:
            scheduler = None
            scheduler_kwargs = {}

        self.optimizer_G = get_base_optimizer(self.g_optim_type)(itertools.chain(*[net.parameters() for net in self.model.NaC_nets]), **self.g_optim_kwargs)
        self.optimizer = BaseDummyOptimizer(optimizer_G=self.optimizer_G, optimizer_D=self.optimizer_D, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs)
        
        self.loss = NotACycleLoss(self.netD, self.model.NaC_nets, self.optimizer_G, self.optimizer_D, self.ndims, **self.loss_kwargs)        
         
    def setup_datapipes(self):
        self.arrays = {}
        self.datapipes = {}
        for id, src in self.sources.items():
            self.datapipes[id] = CycleDataPipe(id, 
                                            src, 
                                            self.ndims, 
                                            self.common_voxel_size, 
                                            self.interp_order, 
                                            self.batch_size)
            self.arrays.update(self.datapipes[id].arrays)

    def make_request(self, mode:str='train'):    
        # create request
        request = gp.BatchRequest()
        for array_name, array in self.arrays.items():
            if (mode == 'prenet' and ('real' in array_name or 'mask' in array_name)) or (mode != 'prenet' and (mode != 'predict' or 'cycle' not in array_name)):
                extents = self.get_extents(array_name=array.identifier)
                request.add(array, self.common_voxel_size * extents, self.common_voxel_size)
        return request

if __name__ == '__main__':
    system = NotACycleGAN(config='./train_conf.json')
    system.logger.info('CycleGAN system loaded. Training...')
    _ = system.train()
    system.logger.info('Done training!')
