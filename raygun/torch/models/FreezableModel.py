import torch

from raygun.torch.models import BaseModel
from raygun.torch.networks.utils import *
from raygun.utils import passing_locals

class FreezableModel(BaseModel):
    def __init__(self, freeze_norms_at=None, **kwargs):        
        super().__init__(**passing_locals(locals()))
    
    def set_norm_modes(self, mode:str='train'):
        for net in self.nets:
            set_norm_mode(net, mode)

    def add_log(self, writer, step):
        means = []
        vars = []
        for net in self.nets:
            mean, var = get_running_norm_stats(getattr(self, net))
            means.append(mean)
            vars.append(var)        
            
        hists = {"means": torch.cat(means), "vars": torch.cat(vars)}
        for tag, values in hists:
            writer.add_histogram(tag, values, global_step=step)
    
    def update_status(self, step):
        if self.freeze_norms_at is not None and step >= self.freeze_norms_at:
            self.set_norm_modes(mode='fix_norms')            
