import torch

from raygun.torch.models import BaseModel
from raygun.torch.networks import UNet
from raygun.torch.networks.UNet import ConvPass


class CAREModel(BaseModel):
    def __init__(self,
                 ndims=2,
                 unet_kwargs={
                                "input_nc": 1,
                                "ngf": 12,
                                "fmap_inc_factor": 6,
                                "downsample_factors": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
                                "constant_upsample": True,
                            },) -> None:
        super.__init__()
        
        # init unet
        self.unet = UNet(**unet_kwargs)
        
        # init model 
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            self.unet,
            ConvPass(unet_kwargs['ngf'], 1, [(1,)*ndims], activation=None),
            torch.nn.Sigmoid()
        )
        
        # setup output data arrays & dict
        self.output_arrays: list[str] = ['pred_affs']
        self.data_dict: dict = {}
        
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
        z = self.unet(raw)
        affs = self.aff_head(z)

        return affs
        