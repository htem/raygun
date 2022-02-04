import torch

class CycleGAN_Model(torch.nn.Module):
    def __init__(self, netG1, netD1, netG2, netD2):
        super(CycleGAN_Model, self).__init__()
        self.netG1 = netG1
        self.netD1 = netD1
        self.netG2 = netG2
        self.netD2 = netD2
        self.cycle = True
    
    def sampling_bottleneck(self, array, scale_factor):
        #TODO: User torch.nn.functional.interpolate(mode='trilnear or bilinear', align_corners=True)
        ...

    def forward(self, real_A=None, real_B=None): 
        if real_A is not None: #allow calling for single direction pass (i.e. prediction)
            fake_B = self.netG1(real_A)
            if self.cycle:                
                cycled_A = self.netG2(fake_B)
            else:
                cycled_A = None
        else:
            fake_B = None
            cycled_A = None
        if real_B is not None:
            fake_A = self.netG2(real_B)
            if self.cycle:
                cycled_B = self.netG1(fake_A)
            else:
                cycled_B = None
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A
