from funlib.learn.torch.models import UNet, ConvPass
import torch
import logging

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO)

# long range affs - use 20 output features
# increase number of downsampling layers for more features in the bottleneck
class MtlsdModel(torch.nn.Module):
    def __init__(self,
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factor=6,
            downsample_factors=[(2,2,2), (2,2,2), (2,2,2)],
            constant_upsample=True,
            num_affs=3
                ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            constant_upsample=constant_upsample,
        )

        self.lsd_head = ConvPass(num_fmaps, 10, [[1,1,1]], activation="Sigmoid")
        self.aff_head = ConvPass(num_fmaps, num_affs, [[1,1,1]], activation="Sigmoid")

    def forward(self, input):

        z = self.unet(input)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds, affs
        