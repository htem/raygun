import torch
import torch.nn.functional as F

class SliceFill_Model(torch.nn.Module):
    def __init__(self, Gnet, norm=torch.nn.BatchNorm3d(num_features=1, momentum=None, affine=False)):
        super(SliceFill_Model, self).__init__()
        self.Gnet = Gnet
        self.norm = norm

    def forward(self, input): 
        # input is array of dims [batch, z (should be 3), y, x]
        input = self.norm(input.unsqueeze(1).float()) # input normalized into tensor of dims [batch, channel, z, y, x]
        adj_slices = input[:,:,[0,2],:,:].squeeze(1) # separate slice data in channel dimension 
        # back to 4D: [b, c, y, x]
        real_mid_slice = input[:,:,1,:,:] # slice to predict

        pred_mid_slice = self.Gnet(adj_slices) # predict middle slice

        return adj_slices, real_mid_slice, pred_mid_slice
