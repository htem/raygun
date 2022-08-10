from raygun.torch.models import FreezableModel
import torch.nn.functional as F

class CycleModel(FreezableModel):
    def __init__(self, netG1, netG2, scale_factor_A=None, scale_factor_B=None, split=False, **kwargs):
        super().__init__(**locals())
        self.cycle = True
        self.crop_pad = None #TODO: Determine if this is depracated
        self.output_arrays = ['fake_B', 'cycled_B', 'fake_A', 'cycled_A']
        self.nets = [netG1, netG2]
    
    def sampling_bottleneck(self, array, scale_factor):
        size = array.shape[-len(scale_factor):]
        mode = {2: 'bilinear', 3: 'trilinear'}[len(size)]
        down = F.interpolate(array, scale_factor=scale_factor, mode=mode, align_corners=True)
        return F.interpolate(down, size=size, mode=mode, align_corners=True)
    
    def set_crop_pad(self, crop_pad, ndims):
        self.crop_pad = (slice(None,None,None),)*2 + (slice(crop_pad,-crop_pad),)*ndims 

    def forward(self, real_A=None, real_B=None): 
        self.real_A = real_A
        self.real_B = real_B
        
        if real_A is not None: #allow calling for single direction pass (i.e. prediction)
            self.fake_B = self.netG1(real_A)
            if self.crop_pad is not None: 
                self.fake_B = self.fake_B[self.crop_pad]
            if self.scale_factor_B: self.fake_B = self.sampling_bottleneck(self.fake_B, self.scale_factor_B) #apply sampling bottleneck
            if self.cycle:                
                if self.split:
                    self.cycled_A = self.netG2(self.fake_B.detach()) # detach to prevent backprop to first generator
                else:
                    self.cycled_A = self.netG2(self.fake_B)
                if self.crop_pad is not None: 
                    self.cycled_A = self.cycled_A[self.crop_pad]
            else:
                self.cycled_A = None
        else:
            self.fake_B = None
            self.cycled_A = None
            
        if real_B is not None:
            self.fake_A = self.netG2(real_B)
            if self.crop_pad is not None: 
                self.fake_A = self.fake_A[self.crop_pad]
            if self.scale_factor_A: self.fake_A = self.sampling_bottleneck(self.fake_A, self.scale_factor_A) #apply sampling bottleneck
            if self.cycle:
                if self.split:
                    self.cycled_B = self.netG1(self.fake_A.detach()) # detach to prevent backprop to first generator
                else:
                    self.cycled_B = self.netG1(self.fake_A)
                if self.crop_pad is not None: 
                    self.cycled_B = self.cycled_B[self.crop_pad]
            else:
                self.cycled_B = None
        else:
            self.fake_A = None
            self.cycled_B = None

        return self.fake_B, self.cycled_B, self.fake_A, self.cycled_A
