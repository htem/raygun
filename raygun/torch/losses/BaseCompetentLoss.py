import torch
from raygun.utils import passing_locals
from raygun.torch.losses import GANLoss

class BaseCompetentLoss(torch.nn.Module):
    def __init__(self,
                **kwargs
                ):
        super().__init__()
        kwargs = passing_locals(locals())
        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, 'gan_mode'):
            self.gan_loss = GANLoss(gan_mode=self.gan_mode)

        self.loss_dict = {}

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

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
                
    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def add_log(self, writer, step):
        for key, loss in self.loss_dict.items():
            writer.add_scalar(key, loss, step)
    
    def update_status(self, step):
        pass