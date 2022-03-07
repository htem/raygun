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
    def __init__(self, Gnet, lowlim_var=1e-5):
        super(SliceFill_Uncertain_Model, self).__init__()
        self.Gnet = Gnet
        self.lowlim_var = lowlim_var

    def forward(self, input): # input is array of dims [batch, z (should be 3), y, x]        
        adj_slices = input.float()[:,[0,2],:,:] # separate slice data in channel dimension (i.e. [b, c, y, x])

        pred_mid_slice = self.Gnet(adj_slices) # predict middle slice [mean, ~variance] in channel dimension
        #predicted mean:
        pred = torch.cat([adj_slices[:,0,:,:].unsqueeze(1), pred_mid_slice[:,0,:,:].unsqueeze(1), adj_slices[:,1,:,:].unsqueeze(1)], 1)
        #predicted variance (tanh outputs need to be scaled and shifted to [0,1] and then clipped to lowlim_var)
        pred_var = torch.clamp(pred_mid_slice[:,1,:,:].unsqueeze(1) * 0.5 + 0.5, self.lowlim_var)
        return pred,  pred_var # predicted variance
