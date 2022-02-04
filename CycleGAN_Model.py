import torch

class CycleGAN_Model(torch.nn.Module):
    def __init__(self, netG1, netD1, netG2, netD2, valid_fake_size=None):
        super(CycleGAN_Model, self).__init__()
        self.netG1 = netG1
        self.netD1 = netD1
        self.netG2 = netG2
        self.netD2 = netD2
        self.valid_fake_size = valid_fake_size
        if valid_fake_size:
            self.dims = len(valid_fake_size)
        self.cycle = True

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]
    
    def sampling_bottleneck(self, array, scale_factor):
        #TODO: User torch.nn.functional.interpolate(mode='trilnear or bilinear', align_corners=True)
        ...

    def forward(self, real_A=None, real_B=None): 
        if real_A is not None: #allow calling for single direction pass (i.e. prediction)
            fake_B = self.netG1(real_A)
            if self.cycle:
                if self.valid_fake_size:
                    fake_B = self.crop(fake_B, self.valid_fake_size)
                cycled_A = self.netG2(fake_B)
            else:
                cycled_A = None
        else:
            fake_B = None
            cycled_A = None
        if real_B is not None:
            fake_A = self.netG2(real_B)
            if self.cycle:
                if self.valid_fake_size:
                    fake_A = self.crop(fake_A, self.valid_fake_size)
                cycled_B = self.netG1(fake_A)
            else:
                cycled_B = None
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A
