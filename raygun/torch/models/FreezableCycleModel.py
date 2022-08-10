import torch.nn.functional as F

from raygun.torch.models import CycleModel
from raygun.torch.networks.utils import *

class FreezableCycleModel(CycleModel):
    def __init__(self, netG1, netG2, scale_factor_A=None, scale_factor_B=None, split=False):
        super().__init__(**locals())
    
    def set_norm_modes(self, mode:str='train', nets= ['netG1', 'netG2']):
        for net in nets:
            set_norm_mode(getattr(self, net), mode)

    def add_log(self, writer, iter, nets= ['netG1', 'netG2']):
        means = []
        vars = []
        for net in nets:
            mean, var = get_running_norm_stats(getattr(self, net))
            means.append(mean)
            vars.append(var)        
        hists = {"means": torch.cat(means), "vars": torch.cat(vars)}
        for tag, values in hists:
            writer.add_histogram(tag, values, global_step=iter)