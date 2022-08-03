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

torch.cuda.set_device(1)
# %%
class Test():
    def __init__(self, net=None, norm=None, size=24, seed=42, noise_factor=3, img='astronaut', ind=31):
        if net is None:
            if norm is None:
                norm = partial(torch.nn.InstanceNorm2d, track_running_stats=True, momentum=0.01)
            self.net = torch.nn.Sequential(
                            ResnetGenerator(1, 1, 32, norm, n_blocks=4), 
                            torch.nn.Sigmoid()
                        ).to('cuda')
        else:
            self.net = net
        self.size = size
        self.mode = 'train'
        self.ind = ind
        self.noise_factor = noise_factor
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters())#, lr=1e-6)
        torch.manual_seed(seed)
        if img is not None:
            self.data = getattr(data, img)()
            if len(self.data.shape) > 2:
                self.data = self.data[...,0]
            self.data = torch.cuda.FloatTensor(self.data).unsqueeze(0) / 255
            self.size = self.data.shape[-1]
    
    def get_norm_layers(self):
        return [n for n in self.net.modules() if 'norm' in type(n).__name__.lower()]

    def get_running_norm_stats(self):
        means = []
        vars = []
        for norm in self.get_norm_layers():
            means.append(norm.running_mean)
            vars.append(norm.running_var)
        return means, vars

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'fix_stats':
            self.net.train()
            for m in self.net.modules():
                if 'norm' in type(m).__name__.lower():
                    m.eval()
        
        if mode == 'train':
            self.net.train()

        if mode == 'eval':
            self.net.eval()
    
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
        if self.data is None:
            ind = torch.randint(low=0, high=200, size=(1,))[0]
            is_face = ind >= 100
            gt = torch.cuda.FloatTensor(data.lfw_subset()[ind][:self.size, :self.size]).unsqueeze(0)
            
        else:
            is_face = None
            gt = self.data
                
        noise = torch.randperm(self.size**2, device='cuda').reshape((self.size, self.size)).unsqueeze(0) / self.size**2 # mean should always be 0.5
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
model = Test()
#%%
patches, gt, out, is_face = model.forward()
model.show()

#%%
model.step(True)

#%%
steps = 1000
show_every = 100
losses = np.zeros((2*steps,))
ticker = trange(steps)
model.set_mode('train')
for step in ticker:
    losses[step] = model.step((step % show_every)==0)
    ticker.set_postfix({'loss':losses[step]})
plt.figure()
plt.plot(losses)

#%%))
ticker = trange(steps)
model.set_mode('fix_stats')
for step in ticker:
    losses[step] = model.step((step % show_every)==0)
    ticker.set_postfix({'loss':losses[step]})
plt.figure()
plt.plot(losses)

#%%
model_allTrain = Test()
model_allFix = Test()
model_switch = Test()

steps = 1000
show_every = 500
allTrain_losses = np.zeros((2*steps,))
allFix_losses = np.zeros((2*steps,))
switch_losses = np.zeros((2*steps,))
ticker = trange(steps)
model_allTrain.set_mode('train')
model_allFix.set_mode('fix_stats')
model_switch.set_mode('train')
for step in ticker:
    allTrain_losses[step] = model_allTrain.step((step % show_every)==0)
    allFix_losses[step] = model_allFix.step((step % show_every)==0)
    switch_losses[step] = model_switch.step((step % show_every)==0)
    ticker.set_postfix({'allTrain':allTrain_losses[step],
                        'allFix': allFix_losses[step],
                        'switch': switch_losses[step]})
plt.figure()
plt.plot(allTrain_losses, label='allTrain')
plt.plot(allFix_losses, label='allFix')
plt.plot(switch_losses, label='switch')
plt.legend()

ticker = trange(steps, steps*2)
model_switch.set_mode('fix_stats')
for step in ticker:
    allTrain_losses[step] = model_allTrain.step((step % show_every)==0)
    allFix_losses[step] = model_allFix.step((step % show_every)==0)
    switch_losses[step] = model_switch.step((step % show_every)==0)
    ticker.set_postfix({'allTrain':allTrain_losses[step],
                        'allFix': allFix_losses[step],
                        'switch': switch_losses[step]})

plt.figure()
plt.plot(allTrain_losses, label='allTrain')
plt.plot(allFix_losses, label='allFix')
plt.plot(switch_losses, label='switch')
plt.legend()
# %%
model_noNorm = Test(norm=torch.nn.Identity)