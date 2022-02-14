import torch

class SliceFill_CARE_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G):
        super(SliceFill_CARE_Optimizer, self).__init__()
        self.optimizer_G = optimizer_G

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass

class SliceFill_ConditionalGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G, optimizer_D):
        super(SliceFill_ConditionalGAN_Optimizer, self).__init__()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass