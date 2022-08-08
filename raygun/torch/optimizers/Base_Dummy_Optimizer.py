import torch

class Base_Dummy_Optimizer(torch.nn.Module):
    def __init__(self, **optimizers):
        super().__init__()
        for name, optimizer in optimizers.items():
            setattr(self, name, optimizer)

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass