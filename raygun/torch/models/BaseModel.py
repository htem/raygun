import torch

class BaseModel(torch.nn.Module):
    def __init__(self) -> None:
        self.output_arrays = []
    
    def add_log(self, writer, iter):        
        pass

    def forward(self):
        return