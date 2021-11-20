from unet import *    
import torch
from gunpowder import Coordinate

input_size = (1,1,240,240)

common_voxel_size=Coordinate((10,10,10))
downsample_factor=2
num_fmaps=24
fmap_inc_factor=2
constant_upsample=False
ndims = 2
n_conv_passes = 3
gnet_depth = 5
kernel_size = 7


kernel_size_down = [[(kernel_size,)*ndims]*n_conv_passes] * gnet_depth
kernel_size_up = [[(kernel_size,)*ndims]*n_conv_passes] * (gnet_depth - 1)


unet = UNet(
                in_channels=1,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=[(downsample_factor,)*ndims,] * (gnet_depth - 1),
                padding='same',
                constant_upsample=constant_upsample,
                voxel_size=common_voxel_size[-ndims:],
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up
                )

netG1 = torch.nn.Sequential(
                    unet,
                    ConvPass(num_fmaps, 1, [(1,)*ndims], activation=None, padding='same'), 
                    torch.nn.Sigmoid())

test = netG1(torch.rand(*input_size))
print(f'In shape: {input_size}')
print(f'Out shape: {test.shape}')

