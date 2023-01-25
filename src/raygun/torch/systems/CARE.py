from distutils.command.config import config
import imp
import itertools
import logging
from matplotlib import pyplot as plt
import torch
import daisy
import gunpowder as gp
import numpy as np

# from raygun.io import CAREDataPipe
# from raygun.torch.models import CAREModel
# from raygun.torch.losses import * TODO: can use ADAM
from raygun.torch.optimizers import BaseDummyOptimizer, get_base_optimizer
from raygun.torch.systems import BaseSystem # For intheritance & subclassing

class CARE(BaseSystem):
    def __init__(self, congfig=None):
        super().__init__(
            default_config='../default_configs/default_CARE_conf.json',
            config=config
        )
        
        # Setup funcs
        self.logger = logging.Logger(__name__, "INFO")
        if self.common_voxel_size is None:
            self.common_voxel_size = gp.Coordinate(
                daisy.open_ds(
                    self.sources["B"]["path"], self.sources["B"]["name"]
                ).voxel_size
            )
        else:
            self.common_voxel_size = gp.Coordinate(self.common_voxel_size)
        if self.ndims is None:
            self.ndims = sum(np.array(self.common_voxel_size) == np.min(self.common_voxel_size))
        
        
    def batch_show(self):
        pass

    