import torch

class SliceFill_Model(torch.nn.Module):
    def __init__(self, Gnet):
        super(SliceFill_Model, self).__init__()
        self.Gnet = Gnet

    def forward(self, input): # input is array of dims [batch, z (should be 3), y, x]        
        adj_slices = input.float()[:,[0,2],:,:] # separate slice data in channel dimension (i.e. [b, c, y, x])

        pred_mid_slice = self.Gnet(adj_slices) # predict middle slice
        pred = torch.cat([adj_slices[:,0,:,:].unsqueeze(1), pred_mid_slice, adj_slices[:,1,:,:].unsqueeze(1)], 1)

        return pred

class SliceFill_Uncertain_Model(torch.nn.Module):
    def __init__(self, Gnet):
        super(SliceFill_Uncertain_Model, self).__init__()
        self.Gnet = Gnet

    def forward(self, input): # input is array of dims [batch, z (should be 3), y, x]        
        adj_slices = input.float()[:,[0,2],:,:] # separate slice data in channel dimension (i.e. [b, c, y, x])

        pred_mid_slice = self.Gnet(adj_slices) # predict middle slice [mean, variance] in channel dimension
        #predicted mean:
        pred = torch.cat([adj_slices[:,0,:,:].unsqueeze(1), pred_mid_slice[:,0,:,:].unsqueeze(1), adj_slices[:,1,:,:].unsqueeze(1)], 1)
        pred_var = torch.sigmoid(pred_mid_slice[:,1,:,:]).unsqueeze(1)
        return pred,  pred_var # predicted variance
