from typing_extensions import final
from unet import *    
import torch
from gunpowder import Coordinate
import numpy as np

input_size = (1,1,240,240)

common_voxel_size=Coordinate((10,10,10))
downsample_factor=2

num_fmaps=16
fmap_inc_factor=3
constant_upsample=True
ndims = 2
n_conv_passes = 2
gnet_depth = 4
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




##########################################
# for side_length in np.arange(240, 512):
#     try:
#         result_size = netG1(torch.rand(1,1,side_length,side_length)).shape
#         print(f'Side length {side_length} successful on first pass, with result side length {result_size[-1]}.')
#         final_size = netG1(torch.rand(1,1,result_size[-1],result_size[-1])).shape
#         print(f'Side length {side_length} successful on both passes, with final side length {final_size[-1]}.')
#     except:
#         print(f'Side length {side_length} failed.')

