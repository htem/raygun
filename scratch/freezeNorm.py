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
    def __init__(self, 
                net=None, 
                activation=None,
                norm=None, 
                size=24, 
                seed=42, 
                noise_factor=3, 
                img='astronaut', 
                ind=31, 
                name=''):
        torch.manual_seed(seed)
        if net is None:
            if norm is None:
                norm = partial(torch.nn.InstanceNorm2d, track_running_stats=True, momentum=0.01)
            if activation is None:
                activation = torch.nn.ReLU
            self.net = torch.nn.Sequential(
                            ResnetGenerator(1, 1, 32, norm, n_blocks=4, activation=activation), 
                            torch.nn.Tanh()
                        ).to('cuda')
        else:
            self.net = net
        self.size = size
        self.mode = 'train'
        self.ind = ind
        self.name = name
        self.noise_factor = noise_factor
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-5)
        if img is not None:
            self.data = getattr(data, img)()
            if len(self.data.shape) > 2:
                self.data = self.data[...,0]
            self.data = (torch.cuda.FloatTensor(self.data).unsqueeze(0) / 255) * 2 - 1
            self.size = self.data.shape[-1]
    
    def get_norm_layers(self):
        return [n for n in self.net.modules() if 'norm' in type(n).__name__.lower()]

    def get_running_norm_stats(self):
        means = []
        vars = []
        for norm in self.get_norm_layers():
            means.append(norm.running_mean)
            vars.append(norm.running_var)
        means = torch.cat(means)
        vars = torch.cat(vars)
        return means, vars

    def set_mode(self, mode=None):
        if mode is None:
            mode = self.mode
        else:
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
    
    def toggle_stat_fix(self):
        if self.mode == 'fix_stats':
            self.set_mode('train')
        else:
            self.set_mode('fix_stats')
    
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
            gt = torch.cuda.FloatTensor(data.lfw_subset()[ind][:self.size, :self.size]).unsqueeze(0) * 2 - 1
            
        else:
            is_face = None
            gt = self.data
                
        noise = ((torch.randperm(self.size**2, device='cuda').reshape((self.size, self.size)).unsqueeze(0) / self.size**2) * 2 - 1).requires_grad_() # should always be mean=0 var=1
        # noise = torch.rand_like(gt, device='cuda', requires_grad=True)

        img = (gt*noise) / self.noise_factor + (gt / self.noise_factor)

        return self.im2batch(img.detach()), self.im2batch(gt), is_face

    def eval(self, show=True, patches=None, gt=None):
        self.net.eval()
        patches, gt, out, is_face = self.forward(patches=patches, gt=gt)
        if show:
            self.show()
        self.set_mode()
        return self.out

    def show(self):
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(self.img, cmap='gray', vmin=-1, vmax=1)
        axs[0].set_ylabel(self.name)
        axs[0].set_title('Input')
        axs[1].imshow(self.out, cmap='gray', vmin=-1, vmax=1)
        axs[1].set_title('Output')
        axs[2].imshow(self.gt, cmap='gray', vmin=-1, vmax=1)
        axs[2].set_title('Actual')

    def forward(self, patches=None, gt=None, is_face=None):
        if patches is None or gt is None:
            patches, gt, is_face = self.get_data()
        self.img = self.batch2im(patches)
        self.gt = self.batch2im(gt)
        out = self.net(patches)
        self.out = self.batch2im(out)
        self.is_face = is_face

        return patches, gt, out, is_face

    def step(self, show=False, patches=None, gt=None):
        self.optim.zero_grad(True)
        patches, gt, out, is_face = self.forward(patches=patches, gt=gt)
        loss = self.loss_fn(out, gt)
        loss.backward()
        self.optim.step()
        if show:
            self.show()
        return loss.item()

def eval_models(data_src, models):
    outs = {}
    patches, gt, is_face = data_src.get_data()
    for name, model in models.items():
        outs[name] = model.eval(show=False, patches=patches, gt=gt)
    num = len(models.keys()) + 2
    fig, axs = plt.subplots(1, num, figsize=(5*num, 5))
    axs[0].imshow(data_src.batch2im(patches), cmap='gray', vmin=-1, vmax=1)
    axs[0].set_title('Input')
    gt = data_src.batch2im(gt)
    axs[-1].imshow(gt, cmap='gray', vmin=-1, vmax=1)
    axs[-1].set_title('Real')
    for ax, name in zip(axs[1:-1], models.keys()):
        ax.imshow(outs[name], cmap='gray', vmin=-1, vmax=1)
        mse = torch.mean((gt - outs[name])**2)
        ax.set_title(f'{name}: MSE={mse}')

#%%
model = Test()
patches, gt, out, is_face = model.forward()
model.show()
model.step(True)

#%%
model_kwargs = {
                # 'activation': torch.nn.SELU
                }

model_names = ['allTrain',
            'allFix',
            'switch_10',
            'switch_200',
            'switch_500',
            'noNorm',
            'noTrack']

models = {}
for name in model_names:
    these_kwargs = model_kwargs.copy()
    these_kwargs['name'] = name
    if name == 'noNorm':
        these_kwargs['norm'] = torch.nn.Identity
    elif name == 'noTrack':
        model_kwargs['norm'] = torch.nn.InstanceNorm2d
    models[name] = Test(**these_kwargs)

steps = 1000
show_every = steps*2
losses = {}
means = np.zeros((steps,))
vars = np.zeros((steps,))
for name in model_names:
    losses[name] = np.zeros((steps,))
    
ticker = trange(steps)
models['allFix'].set_mode('fix_stats')
data_src = Test()
for step in ticker:
    ticker_postfix = {}
    patches, gt, is_face = data_src.get_data()
    for name, model in models.items():
        if 'switch' in name:
            if (step % int(name.split('_')[-1])) == 0 and step > 0:
                model.toggle_stat_fix()
        losses[name][step] = model.step((step % show_every)==0, patches=patches, gt=gt)
        ticker_postfix[name] = losses[name][step]
    tempM, tempV = models['allTrain'].get_running_norm_stats()
    means[step], vars[step] = tempM.mean(), tempV.mean()
    ticker.set_postfix(ticker_postfix)

#%%
eval_models(data_src, models)
# tempM, tempV = models['switch'].get_running_norm_stats()
# print(f'For Switch training: Mean mean: {tempM.mean()}, Mean var: {tempV.mean()}')

#%%
plt.figure(figsize=(15,10))
for name, loss in losses.items():
    plt.plot(loss, label=name)
plt.title('Losses')
plt.ylim([0,.1])
plt.legend()
 # %%
plt.figure(figsize=(15,10))
plt.plot(means, label='Means')
plt.plot(vars, label='Variances')
plt.legend()
# %%
