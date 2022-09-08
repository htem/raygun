from raygun.torch.models import FreezableModel
from raygun.utils import passing_locals
import torch.nn.functional as F
import torch

# class NotACycleModel(FreezableModel):
class NotACycleModel(nn.Module):
    def __init__(self, 
                **kwargs
                ):        
        output_arrays = ['fake_B', 'cycled_B', 'fake_A', 'cycled_A']
        nets = [netG1, netG2]
        super().__init__(**passing_locals(locals()))

        self.crop_pad = None #TODO: Determine if this is depracated
    
    def set_crop_pad(self, crop_pad, ndims):
        self.crop_pad = (slice(None,None,None),)*2 + (slice(crop_pad,-crop_pad),)*ndims

    def forward(self, real_A=None, real_B=None): 
        assert real_A is not None or real_B is not None, 'Must have some real input to generate outputs)'
        if real_A is not None: #allow calling for single direction pass (i.e. prediction)
            fake_B = self.netG1(real_A)
            if self.crop_pad is not None: 
                fake_B = fake_B[self.crop_pad]
            if self.scale_factor_B: fake_B = self.sampling_bottleneck(fake_B, self.scale_factor_B) #apply sampling bottleneck
            if self.cycle:                
                if self.split:
                    cycled_A = self.netG2(fake_B.detach()) # detach to prevent backprop to first generator
                else:
                    cycled_A = self.netG2(fake_B)
                if self.crop_pad is not None: 
                    cycled_A = cycled_A[self.crop_pad]
            else:
                cycled_A = None
        else:
            fake_B = None
            cycled_A = None
            
        if real_B is not None:
            fake_A = self.netG2(real_B)
            if self.crop_pad is not None: 
                fake_A = fake_A[self.crop_pad]
            if self.scale_factor_A: fake_A = self.sampling_bottleneck(fake_A, self.scale_factor_A) #apply sampling bottleneck
            if self.cycle:
                if self.split:
                    cycled_B = self.netG1(fake_A.detach()) # detach to prevent backprop to first generator
                else:
                    cycled_B = self.netG1(fake_A)
                if self.crop_pad is not None: 
                    cycled_B = cycled_B[self.crop_pad]
            else:
                cycled_B = None
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A

