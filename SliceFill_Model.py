import torch
import torch.nn.functional as F

class SliceFill_Model(torch.nn.Module):
    def __init__(self, Gnet, norm=torch.nn.BatchNorm3d(num_features=1, momentum=None, affine=False)):
        super(SliceFill_Model, self).__init__()
        self.Gnet = Gnet
        self.norm = norm

    def forward(self, input): # input is array of dims [batch, z (should be 3), y, x]        
        norm_real = self.norm(input.unsqueeze(1).float()) # input normalized into tensor of dims [batch, channel, z, y, x]
        norm_real = norm_real.squeeze(1) # make volume for Dnet to adjudicate by concatenating slices in channel dimension        
        adj_slices = norm_real[:,[0,2],:,:] # separate slice data in channel dimension (i.e. [b, c, y, x])

        pred_mid_slice = self.Gnet(adj_slices) # predict middle slice
        pred = torch.cat([adj_slices[:,0,:,:].unsqueeze(1), pred_mid_slice, adj_slices[:,1,:,:].unsqueeze(1)], 1)

        return norm_real, pred
