# Below is an example of a network ("netG1") that will take in a 256x256 tile and produce a 252x252 tile, despite using "same" padding

from funlib.learn.torch.models.unet import UNet, ConvPass
# from unet import *    # <===== Fixed issue 2021/11/17
import torch
from gunpowder import Coordinate

common_voxel_size=Coordinate((10,10,10))
downsample_factor=2
num_fmaps=16
fmap_inc_factor=2
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

test = netG1(torch.rand(1,1,256,256))
print('In shape: (1,1,256,256)')
print(f'Out shape: {test.shape}')



#Testing cropping
print('Testing deepest Upsample level cropping:')
low_tester = torch.rand(1,256,32,32)
upper = netG1[0].r_up[0][0]
low_test = upper.up(low_tester)
low_cropped = upper.crop_to_factor(low_test, upper.crop_factor, upper.next_conv_kernel_sizes)
print(f'Shape after upsample, before crop: {low_test.shape}')
print(f'Shape after crop: {low_cropped.shape}')

