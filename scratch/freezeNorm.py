#%%
from functools import partial
import numpy as np
import torch
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from utils import *
from skimage import data
import matplotlib.pyplot as plt
from tqdm import trange
# %%
class Test():
    def __init__(self, net, size=24, seed=42, noise_factor=3):
        self.net = net
        self.size = size
        self.mode = 'stats'
        self.noise_factor = noise_factor
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-6)
        torch.manual_seed(seed)
    
    def im2batch(self, img):
        mid = self.size // 2        
        patches = []
        patches.append(torch.cuda.FloatTensor(img[:, :mid, :mid]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, mid:, :mid]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, :mid, mid:]).unsqueeze(0))
        patches.append(torch.cuda.FloatTensor(img[:, mid:, mid:]).unsqueeze(0))
        return torch.cat(patches).requires_grad_()
    
    def batch2im(self, batch):
        batch = batch.detach().cpu().squeeze()
        return torch.cat((torch.cat((batch[0], batch[1])), torch.cat((batch[2], batch[3]))), axis=1)

    def get_data(self):
        noise = torch.randperm(self.size**2, device='cuda').reshape((self.size, self.size)).unsqueeze(0) / self.size**2

        ind = 31 # torch.randint(low=0, high=200, size=(1,))[0]
        is_face = ind >= 100
        gt = torch.cuda.FloatTensor(data.lfw_subset()[ind][:self.size, :self.size]).unsqueeze(0)
        # noise = torch.rand_like(gt, device='cuda', requires_grad=True)
        
        img = (gt*noise) / self.noise_factor + (gt / self.noise_factor)

        return self.im2batch(img), self.im2batch(gt), is_face

    def show(self):
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(self.img, cmap='gray')
        axs[0].set_title('Input')
        axs[1].imshow(self.out, cmap='gray')
        axs[1].set_title('Output')
        axs[2].imshow(self.gt, cmap='gray')
        axs[2].set_title('Actual')

    def forward(self):
        patches, gt, is_face = self.get_data()
        self.img = self.batch2im(patches)
        self.gt = self.batch2im(gt)
        out = self.net(patches)
        self.out = self.batch2im(out)
        self.is_face = is_face

        return patches, gt, out, is_face

    def step(self, show=False):
        patches, gt, out, is_face = self.forward()
        loss = self.loss_fn(out, gt)
        loss.backward()
        self.optim.step()
        if show:
            self.show()
        return loss.item()

#%%

norm = partial(torch.nn.InstanceNorm2d, track_running_stats=True, momentum=0.01)
net = torch.nn.Sequential(
            ResnetGenerator(1, 1, 32, norm, n_blocks=4), 
            torch.nn.Sigmoid()
            ).to('cuda')

model = Test(net)
#%%
patches, gt, out, is_face = model.forward()
model.show()

#%%
model.step(True)

#%%
steps = 1000
show_every = 100
losses = np.zeros((steps,))
ticker = trange(steps)
model.net.train()
for step in ticker:
    losses[step] = model.step((step % show_every)==0)
    ticker.set_postfix({'loss':losses[step]})
plt.figure()
plt.plot(losses)

# fixed_norm_layers = []
# for norm_layer in [norm_layers]:
#     # norm_layer.eval()
#     # norm_layer.parameters.freeze()
#     # norm_layer.no_grad()
#     # norm_layer.requires_grad(False)
#     fixed_norm_layer.append(<layer that divides by variance and subtracts mean from norm_layer.running_stats>) --> replace the laye5rs
# %%
