import torch

class CycleGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G, optimizer_D):
        super(CycleGAN_Optimizer, self).__init__()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass

class Split_CycleGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G1, optimizer_D1, optimizer_G2, optimizer_D2):
        super(Split_CycleGAN_Optimizer, self).__init__()
        self.optimizer_G1 = optimizer_G1
        self.optimizer_D1 = optimizer_D1
        self.optimizer_G2 = optimizer_G2
        self.optimizer_D2 = optimizer_D2

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass