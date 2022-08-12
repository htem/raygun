import torch

class BaseModel(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        for array in self.output_arrays:
            setattr(self, array, torch.Tensor())
    
    def add_log(self, writer, iter):        
        pass

    def forward(self):
        return