#adapted from funlib.learn.torch.models

from funlib.learn.torch.models.conv4d import Conv4d
import math
import numpy as np
import torch
import torch.nn as nn


class ConvPass(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            activation,
            final_activation=True,
            padding='valid',
            residual=True,
            padding_mode='reflect'#default to 'zeros' until 1/28/2022 (Jeff Rhoades)
            ):

        super(ConvPass, self).__init__()

        if activation is not None:
            self.activation = getattr(torch.nn, activation)()
        else:
            self.activation = nn.Identity()

        self.final_activation = final_activation
        self.padding = padding
        self.residual = residual

        layers = []

        for i, kernel_size in enumerate(kernel_sizes):

            self.dims = len(kernel_size)

            conv = {
                2: torch.nn.Conv2d,
                3: torch.nn.Conv3d,
                4: Conv4d
            }[self.dims]

            # if padding == 'same':
            #     pad = tuple(k//2 for k in kernel_size)
            # else:
            #     pad = 0

            try:
                layers.append(
                    conv(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding, 
                        # padding=pad, 
                        padding_mode=padding_mode
                        ))
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            in_channels = out_channels

            if not (i == (len(kernel_sizes) - 1)): #omit activation from final layer to allow for residual addition
                layers.append(self.activation)            

        self.conv_pass = torch.nn.Sequential(*layers)

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

    def forward(self, x):
        out = self.conv_pass(x)
        if self.residual:
            if self.padding == 'valid':                
                out += self.crop(x, out.size()[-self.dims:])
            else:
                out += x

        if self.final_activation:
            return self.activation(out)
        else:
            return out



class Downsample(torch.nn.Module):

    def __init__(
            self,
            downsample_factor,
            flexible=False):

        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor
        self.flexible = flexible

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(
            downsample_factor,
            stride=downsample_factor,
            ceil_mode=True) #ceil_mode added to attempt to increase flexibility

    def forward(self, x):
        if self.flexible:
            try:
                return self.down(x)
            except:
                self.check_mismatch(x.size())
        else:
            self.check_mismatch(x.size())
            return self.down(x)
            
    def check_mismatch(self, size):
        for d in range(1, self.dims + 1):
                    if size[-d] % self.downsample_factor[-d] != 0:
                        raise RuntimeError(
                            "Can not downsample shape %s with factor %s, mismatch "
                            "in spatial dimension %d" % (
                                size,
                                self.downsample_factor,
                                self.dims - d))
        return


class Upsample(torch.nn.Module):

    def __init__(
            self,
            scale_factor,
            mode=None,
            in_channels=None,
            out_channels=None,
            crop_factor=None,
            next_conv_kernel_sizes=None):

        super(Upsample, self).__init__()

        if crop_factor is not None:
            assert next_conv_kernel_sizes is not None, "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes
        self.dims = len(scale_factor)

        if mode == 'transposed_conv':

            up = {
                2: torch.nn.ConvTranspose2d,
                3: torch.nn.ConvTranspose3d
            }[self.dims]

            self.up = up(
                in_channels,
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor)

        else:

            self.up = torch.nn.Upsample(
                scale_factor=scale_factor,
                mode=mode)

    def crop_to_factor(self, x, factor, kernel_sizes):
        '''Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        '''

        shape = x.size()
        spatial_shape = shape[-self.dims:]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes)
            for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c)/f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n*f + c
            for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:

            assert all((
                    (t > c) for t, c in zip(
                        target_spatial_shape,
                        convolution_crop))
                ), \
                "Feature map with shape %s is too small to ensure " \
                "translation equivariance with factor %s and following " \
                "convolutions %s" % (
                    shape,
                    factor,
                    kernel_sizes)

            return self.crop(x, target_spatial_shape)

        return x

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

    def forward(self, g_out):

        g_up = self.up(g_out)

        if self.crop_factor is not None:
            g_cropped = self.crop_to_factor(
                g_up,
                self.crop_factor,
                self.next_conv_kernel_sizes)
        else:
            g_cropped = g_up


        return g_cropped


class ResidualUNet(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            activation='ReLU',
            fov=(1, 1, 1),
            voxel_size=(1, 1, 1),
            num_heads=1,
            constant_upsample=False,
            padding='valid',
            residual=True):
        '''Create a Residual U-Net (with or without residual blocks)::

             /------------------------------------------------------\
            f_in --> f_left --------------------------->> f_right -(+)-> f_out
                        |                                   ^
                        |                                   |
                        v  /--------------------------\     |
                        g_in --> g_left -->> g_right -(+)-> g_out
                                    |            ^
                                    v            |
                                        ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively. Additionally, ``/-------\`` connections indicate a residual 
        skip connections between the inputs of one level on the down side and the 
        output of the convolutional layers on the up flowing side, prior to activation.
        At each level, a 1x1(x1) kernel convolution projects the input to the correct
        dimensional feature space.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        Only works for in_channels == out_channels.
        It will perform 4D convolutions as long as ``length`` is greater than 1.
        As soon as ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            in_channels:

                The number of input channels.

            num_fmaps:

                The number of feature maps in the first layer. Stored in the ``channels``
                dimension.

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            fov (optional):

                Initial field of view in physical units

            voxel_size (optional):

                Size of a voxel in the input data, in physical units

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            padding (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).

            residual (optional):

                Whether, at each level, to train convolutional layers to output residuals to add to level inputs (as in ResNet residual blocks) or to directly convolve input data to output. Either 'True' (default) or 'False'. Note: Skip connections between up and down paths are combined as residuals reqardless.
        '''

        super(ResidualUNet, self).__init__()

        self.ndims = len(downsample_factors[0])
        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.residual = residual
        self.padding = padding
        if activation is not None:
            self.activation = getattr(torch.nn, activation)()
        else:
            self.activation = nn.Identity()
        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3,)*self.ndims, (3,)*self.ndims]]*self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3,)*self.ndims, (3,)*self.ndims]]*(self.num_levels - 1)

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if padding.lower() == 'valid':
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f*ff
                        for f, ff in zip(factor, factor_product))
            elif padding.lower() == 'same':
                factor_product = None
            else:
                raise f'Invalid padding option: {padding}'
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left projection convolutional passes
        self.l_proj = nn.ModuleList([
            ConvPass(
                in_channels
                if level == 0
                else num_fmaps*fmap_inc_factor**(level - 1),
                num_fmaps*fmap_inc_factor**level,
                [np.ones_like(kernel_size_down[level][0])],
                activation=activation,
                final_activation=False,
                padding=padding,
                residual=False)
            for level in range(self.num_levels)
        ])

        # left convolutional passes
        self.l_conv = nn.ModuleList([
            ConvPass(
                num_fmaps*fmap_inc_factor**level,
                num_fmaps*fmap_inc_factor**level,
                kernel_size_down[level],
                activation=activation,
                padding=padding,
                residual=self.residual)
            for level in range(self.num_levels)
        ])
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList([
            Downsample(downsample_factors[level])
            for level in range(self.num_levels - 1)
        ])

        # right up/crop layers
        self.r_up = nn.ModuleList([
            nn.ModuleList([
                Upsample(
                    downsample_factors[level],
                    mode='nearest' if constant_upsample else 'transposed_conv',
                    in_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    out_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    crop_factor=crop_factors[level],
                    next_conv_kernel_sizes=kernel_size_up[level])
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

        # right projection convolutional passes
        self.r_proj = nn.ModuleList([
            nn.ModuleList([
                ConvPass(
                    num_fmaps*fmap_inc_factor**(level + 1),
                    num_fmaps*fmap_inc_factor**level,
                    [np.ones_like(kernel_size_up[level][0])],
                    activation=activation,
                    final_activation=False,
                    padding=padding,
                    residual=False)
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

        # right convolutional passes
        self.r_conv = nn.ModuleList([
            nn.ModuleList([
                ConvPass(
                    num_fmaps*fmap_inc_factor**level,
                    num_fmaps*fmap_inc_factor**level,
                    kernel_size_up[level],
                    activation=activation,
                    final_activation=False,
                    padding=padding,
                    residual=self.residual)
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

        # right final projection convolutional passes
        self.r_fin = nn.ModuleList([
            ConvPass(
                num_fmaps,
                in_channels,
                [np.ones_like(kernel_size_down[self.num_levels - 1][0])],
                activation=activation,
                padding=padding,
                residual=False)
            for _ in range(num_heads)
        ])

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

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.num_levels - level - 1

        # project to feature space
        f_proj = self.l_proj[i](f_in) # no final activation applied

        # convolve
        f_left = self.l_conv[i](f_proj)

        # end of recursion (bottom level)
        if level == 0:

            fs_out = [f_left]*self.num_heads

        else:

            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            # up and crop
            gs_cropped = [
                self.r_up[h][i](gs_out[h])
                for h in range(self.num_heads)
            ]

            fs_right = [ # no final activation applied
                self.r_proj[h][i](gs_cropped[h])
                for h in range(self.num_heads)
            ]

            # convolve
            fs_mid = [ # no final activation applied
                self.r_conv[h][i](fs_right[h])
                for h in range(self.num_heads)
            ]

            # add residuals to identity projections and activate
            if self.padding.lower() == 'valid':
                f_id_cropped = self.crop(f_proj, fs_mid[0].size()[-self.dims:])
            elif self.padding.lower() == 'same':
                f_id_cropped = f_proj
            
            fs_out = [
                self.activation(fs_mid[h] + f_id_cropped)
                for h in range(self.num_heads)
            ]

        return fs_out

    def forward(self, x):
        
        y = [
            self.r_fin[h](self.rec_forward(self.num_levels - 1, x)[h])
            for h in range(self.num_heads)
        ]

        if self.num_heads == 1:
            return y[0]

        return y