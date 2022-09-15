import torch
import functools


class NLayerDiscriminator2D(torch.nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc=1,
        ngf=64,
        n_layers=3,
        norm_layer=torch.nn.BatchNorm2d,
        kw=4,
        downsampling_kw=None,
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ngf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d

        if downsampling_kw is None:
            downsampling_kw = kw

        padw = 1
        ds_kw = downsampling_kw
        sequence = [
            torch.nn.Conv2d(input_nc, ngf, kernel_size=ds_kw, stride=2, padding=padw),
            torch.nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                torch.nn.Conv2d(
                    ngf * nf_mult_prev,
                    ngf * nf_mult,
                    kernel_size=ds_kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ngf * nf_mult),
                torch.nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            torch.nn.Conv2d(
                ngf * nf_mult_prev,
                ngf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ngf * nf_mult),
            torch.nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            torch.nn.Conv2d(ngf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = torch.nn.Sequential(*sequence)

    @property
    def FOV(self):
        # Returns the receptive field of one output neuron for a network (written for patch discriminators)
        # See https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-region for formula derivation

        L = 0  # num of layers
        k = []  # [kernel width at layer l]
        s = []  # [stride at layer i]
        for layer in self.model:
            if hasattr(layer, "kernel_size"):
                L += 1
                k += [layer.kernel_size[-1]]
                s += [layer.stride[-1]]

        r = 1
        for l in range(L - 1, 0, -1):
            r = s[l] * r + (k[l] - s[l])

        return r

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminator3D(torch.nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc=1,
        ngf=64,
        n_layers=3,
        norm_layer=torch.nn.BatchNorm3d,
        kw=4,
        downsampling_kw=None,
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ngf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == torch.nn.InstanceNorm3d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm3d

        if downsampling_kw is None:
            downsampling_kw = kw

        padw = 1
        ds_kw = downsampling_kw
        sequence = [
            torch.nn.Conv3d(input_nc, ngf, kernel_size=ds_kw, stride=2, padding=padw),
            torch.nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                torch.nn.Conv3d(
                    ngf * nf_mult_prev,
                    ngf * nf_mult,
                    kernel_size=ds_kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ngf * nf_mult),
                torch.nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            torch.nn.Conv3d(
                ngf * nf_mult_prev,
                ngf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ngf * nf_mult),
            torch.nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            torch.nn.Conv3d(ngf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminator(NLayerDiscriminator2D, NLayerDiscriminator3D):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ndims, **kwargs):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ngf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        if ndims == 2:
            NLayerDiscriminator2D.__init__(self, **kwargs)
        elif ndims == 3:
            NLayerDiscriminator3D.__init__(self, **kwargs)
        else:
            raise ValueError(
                ndims,
                "Only 2D or 3D currently implemented. Feel free to contribute more!",
            )
