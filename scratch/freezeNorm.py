#%%
from functools import partial
import torch
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from utils import *
from skimage import data
import matplotlib.pyplot as plt
import 

# %%
class Test():
    def __init__(self, net, size=24, seed=42):
        self.net = net
        self.size = size
        self.mode = 'stats'
        torch.manual_seed(seed)
    
    def get_patches(self, img):
        mid = self.size // 2        
        patches = []
        patches.append(torch.cuda.FloatTensor(img[:, :mid, :mid]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, mid:, :mid]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, :mid, mid:]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, mid:, mid:]).unsqueeze(0))
        return patches

    def get_data(self):
        if self.mode == 'stats':
            is_face = None
            img = torch.randperm(self.size**2, device='cuda').reshape((self.size, self.size)).unsqueeze(0) / self.size**2

        elif self.mode == 'train':
            ind = torch.randint(low=0, high=200, size=(1,))[0]
            is_face = ind >= 100
            img = torch.cuda.FloatTensor(data.lfw_subset()[ind][:self.size, :self.size]).unsqueeze(0)

        return self.get_patches(img), img.detach().cpu().squeeze(), is_face

    def show(self):
        fig, axs = plt.subplots(1, 3, figsize=(30,10))
        axs[0].imshow(self.img, cmap='grays')
        axs[1].imshow(self.out, cmap='grays')
        axs[2].imshow(self.img - self.out)

    def forward(self):
        patches, img, is_face = self.get_data()
        self.img = img
        self.is_face = is_face
        outs = []
        for patch in patches:
            outs.append(self.net(patch).detach().cpu().squeeze())        
        self.out = torch.cat((torch.cat((outs[0], outs[1])), torch.cat((outs[2], outs[3]))), axis=1)

        ...#TODO:LOSS
#%%

norm = partial(torch.nn.InstanceNorm2d, track_running_stats=True, momentum=0.01)
net = torch.nn.Sequential(
            ResnetGenerator(1, 1, 16, norm, n_blocks=4), 
            torch.nn.Sigmoid()
            ).to('cuda')

model = Test(net)

#%%
model.forward()
model.show()

#%%
model.mode = 'train'
model.forward()
model.show()

# fixed_norm_layers = []
# for norm_layer in [norm_layers]:
#     # norm_layer.eval()
#     # norm_layer.parameters.freeze()
#     # norm_layer.no_grad()
#     # norm_layer.requires_grad(False)
#     fixed_norm_layer.append(<layer that divides by variance and subtracts mean from norm_layer.running_stats>) --> replace the laye5rs
# %%
