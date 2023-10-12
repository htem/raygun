from raygun.evaluation.validate_affinities import run_validation
import torch

from raygun.utils import passing_locals
from raygun.torch.losses import GANLoss


class BaseCompetentLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = passing_locals(locals())
        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "gan_mode"):
            self.gan_loss = GANLoss(gan_mode=self.gan_mode)

        self.loss_dict = {}

    def set_requires_grad(self, nets:list, requires_grad:bool=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, "weight") and hasattr(module.weight, "data"):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def add_log(self, writer, step):
        # add loss values
        for key, loss in self.loss_dict.items():
            writer.add_scalar(key, loss, step)

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

    def update_status(self, step):
        if hasattr(self, "validation_config") and (
            step % self.validation_config["validate_every"] == 0
        ):
            run_validation(self.validation_config, step)
