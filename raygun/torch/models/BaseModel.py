import torch

class BaseModel(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
                
        assert hasattr(self, 'output_arrays'), "Model object must have list attribute `output_arrays` indicating what arrays are output by the model's forward pass, in order."
    
    def add_log(self, writer, iter):        
        pass

    def forward(self):
        return