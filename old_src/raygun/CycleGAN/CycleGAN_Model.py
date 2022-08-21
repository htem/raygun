import torch
import torch.nn.functional as F

class CycleGAN_Model(torch.nn.Module):
    def __init__(self, netG1, netD1, netG2, netD2, scale_factor_A=None, scale_factor_B=None):
        super(CycleGAN_Model, self).__init__()
        self.netG1 = netG1
        self.netD1 = netD1
        self.netG2 = netG2
        self.netD2 = netD2
        self.scale_factor_A = scale_factor_A
        self.scale_factor_B = scale_factor_B
        self.cycle = True
        self.crop_pad = False
    
    def sampling_bottleneck(self, array, scale_factor):
        size = array.shape[-len(scale_factor):]
        mode = {2: 'bilinear', 3: 'trilinear'}[len(size)]
        down = F.interpolate(array, scale_factor=scale_factor, mode=mode, align_corners=True)
        return F.interpolate(down, size=size, mode=mode, align_corners=True)
    
    def set_crop_pad(self, crop_pad, ndims):
        self.crop_pad = (slice(None,None,None),)*2 + (slice(crop_pad,-crop_pad),)*ndims

    def forward(self, real_A=None, real_B=None): 
        if real_A is not None: #allow calling for single direction pass (i.e. prediction)
            fake_B = self.netG1(real_A)
            if self.crop_pad is not False: 
                fake_B = fake_B[self.crop_pad]
            if self.scale_factor_B: fake_B = self.sampling_bottleneck(fake_B, self.scale_factor_B) #apply sampling bottleneck
            if self.cycle:                
                cycled_A = self.netG2(fake_B)
                if self.crop_pad is not False: 
                    cycled_A = cycled_A[self.crop_pad]
            else:
                cycled_A = None
        else:
            fake_B = None
            cycled_A = None
        if real_B is not None:
            fake_A = self.netG2(real_B)
            if self.crop_pad is not False: 
                fake_A = fake_A[self.crop_pad]
            if self.scale_factor_A: fake_A = self.sampling_bottleneck(fake_A, self.scale_factor_A) #apply sampling bottleneck
            if self.cycle:
                cycled_B = self.netG1(fake_A)
                if self.crop_pad is not False: 
                    cycled_B = cycled_B[self.crop_pad]
            else:
                cycled_B = None
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A

class CycleGAN_Split_Model(torch.nn.Module):
    def __init__(self, netG1, netD1, netG2, netD2, scale_factor_A=None, scale_factor_B=None):
        super(CycleGAN_Split_Model, self).__init__()
        self.netG1 = netG1
        self.netD1 = netD1
        self.netG2 = netG2
        self.netD2 = netD2
        self.scale_factor_A = scale_factor_A
        self.scale_factor_B = scale_factor_B
        self.cycle = True
        self.crop_pad = False
    
    def sampling_bottleneck(self, array, scale_factor):
        size = array.shape[-len(scale_factor):]
        mode = {2: 'bilinear', 3: 'trilinear'}[len(size)]
        down = F.interpolate(array, scale_factor=scale_factor, mode=mode, align_corners=True)
        return F.interpolate(down, size=size, mode=mode, align_corners=True)

    def set_crop_pad(self, crop_pad, ndims):
        self.crop_pad = (slice(None,None,None),)*2 + (slice(crop_pad,-crop_pad),)*ndims
        
    def forward(self, real_A=None, real_B=None): 
        if real_A is not None: #allow calling for single direction pass (i.e. prediction)
            fake_B = self.netG1(real_A)
            if self.crop_pad is not False: 
                fake_B = fake_B[self.crop_pad]
            if self.scale_factor_B: fake_B = self.sampling_bottleneck(fake_B, self.scale_factor_B) #apply sampling bottleneck
            if self.cycle:                
                cycled_A = self.netG2(fake_B.detach()) # detach to prevent backprop to first generator
                if self.crop_pad is not False: 
                    cycled_A = cycled_A[self.crop_pad]
            else:
                cycled_A = None
        else:
            fake_B = None
            cycled_A = None
        if real_B is not None:
            fake_A = self.netG2(real_B)
            if self.crop_pad is not False: 
                fake_A = fake_A[self.crop_pad]
            if self.scale_factor_A: fake_A = self.sampling_bottleneck(fake_A, self.scale_factor_A) #apply sampling bottleneck
            if self.cycle:
                cycled_B = self.netG1(fake_A.detach()) # detach to prevent backprop to first generator
                if self.crop_pad is not False: 
                    cycled_B = cycled_B[self.crop_pad]
            else:
                cycled_B = None
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A
