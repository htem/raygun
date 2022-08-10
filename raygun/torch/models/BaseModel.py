import torch

class BaseModel(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.output_arrays = []
    
    def add_log(self, writer, iter):        
        pass

    def forward(self):
        return