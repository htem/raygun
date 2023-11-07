# from funlib.learn.torch.models import UNet, ConvPass
from raygun.torch.networks import UNet
from raygun.torch.networks.UNet import ConvPass
import torch
import logging

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO)

# long range affs - use 20 output features
# increase number of downsampling layers for more features in the bottleneck
class ACLSDModel(torch.nn.Module):
    def __init__(
        self,
        mt_unet_kwargs={
            "input_nc": 1,
            "ngf": 12,
            "fmap_inc_factor": 6,
            "num_heads": 2,
            "downsample_factors": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            "constant_upsample": True,
            "padding_type": "same",
        },
        ac_unet_kwargs={
            "input_nc": 1,
            "ngf": 12,
            "fmap_inc_factor": 6,
            "downsample_factors": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            "constant_upsample": True,
            "padding_type": "same",
        },
        num_affs=3,
    ):
        super().__init__()

        self.mt_unet = UNet(**mt_unet_kwargs)
        self.ac_unet = UNet(**ac_unet_kwargs)

        self.aff_head = ConvPass(
            mt_unet_kwargs["ngf"], num_affs, [[1, 1, 1]], activation="Sigmoid"
        )

        self.lsd_head = ConvPass(  # TODO: Make work without LSD
            mt_unet_kwargs["ngf"], 10, [[1, 1, 1]], activation="Sigmoid"
        )

        self.ac_aff_head = ConvPass(
            ac_unet_kwargs["ngf"], num_affs, [[1, 1, 1]], activation="Sigmoid"
        )

        self.output_arrays = ["pred_affs", "pred_lsds", "pred_affs_ac"]
        self.data_dict = {}

    def add_log(self, writer, step):
        # add loss input image examples
        for name, data in self.data_dict.items():
            if len(data.shape) > 3:  # pull out batch dimension if necessary
                img = data[0].squeeze()
            else:
                img = data.squeeze()

            if len(img.shape) == 3:
                mid = img.shape[0] // 2  # for 3D volume
                img = img[mid]

            if (
                (img.min() < 0) and (img.min() >= -1.0) and (img.max() <= 1.0)
            ):  # scale img to [0,1] if necessary
                img = (img * 0.5) + 0.5
            writer.add_image(name, img, global_step=step, dataformats="HW")

    def forward(self, raw):
        self.data_dict.update({"raw": raw.detach()})
        a = self.mt_unet(raw)
        # conv passes for MTLSD
        affs = self.aff_head(a)
        lsds = self.lsd_head(a)
        b = self.ac_unet(lsds)
        # conv pass for ACLSD
        affs_ac = self.ac_aff_head(b)

        return affs, lsds, affs_ac
