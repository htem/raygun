#CAUTION UNFINISHED; TODO: Incorporate cropping of input due to UNet


# from unet import *    
import functools
import torch
from tri_utils import NLayerDiscriminator, NLayerDiscriminator3D

input_size = (1,1,240,240)
num_fmaps = 16
dnet_depth = 5
downsample_factor = 2
ndims = 2
kernel_size = 5


#For discriminators:
if ndims == 3: #3D case
    norm_instance = torch.nn.InstanceNorm3d
    discriminator_maker = NLayerDiscriminator3D
elif ndims == 2:
    norm_instance = torch.nn.InstanceNorm2d
    discriminator_maker = NLayerDiscriminator

#For netD1:
norm_layer = functools.partial(norm_instance, affine=False, track_running_stats=False)
netD1 = discriminator_maker(input_nc=1, 
                                ndf=num_fmaps, 
                                n_layers=dnet_depth, 
                                norm_layer=norm_layer,
                                downsampling_kw=downsample_factor, 
                                kw=kernel_size,
                            )

test = netD1(torch.rand(*input_size))
print(f'In shape: {input_size}')
print(f'Out shape: {test.shape}')

