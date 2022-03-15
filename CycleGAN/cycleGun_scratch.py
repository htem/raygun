# %%
from itertools import cycle
import sys
sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220215_ import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220304_ import *
# from CycleGun20220304XNH2EM_apply_cb2SynapseCutout1_ import *
# from SplitCycleGun20220304XNH2EM_apply_cb2SynapseCutout1_ import *
# from SplitCycleGun20220308XNH2EM_apply_cb2SynapseCutout1_ import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220309seluSplitNoise_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220309seluSplit_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220310unetSplitNoise_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220310splitUnetConvDown_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220310unet_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220310validSplitResWasserSeluNoise_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220311LinkResSelu_train import *
# from CycleGun_CBv30nmBottom100um_cb2gcl1_20220311SplitResSelu_train import *
from SplitCycleGun20220311XNH2EM_apply_cb2SynapseCutout1_ import *
import matplotlib.pyplot as plt
import zarr

# %%
batch = cycleGun.test_train()

# %%
batch = cycleGun.test_prediction('B', side_length=200, cycle=True)
batch = cycleGun.test_prediction('A', side_length=200, cycle=True)

# %%
cycleGun.model.eval()
# cycleGun.model.train()

net = cycleGun.model.netG1
real = batch[cycleGun.real_A].data * 2 - 1
# net = cycleGun.model.netG2
# real = batch[cycleGun.real_B].data * 2 - 1

mid = real.shape[-1] // 2
test = net(torch.cuda.FloatTensor(real).unsqueeze(0))
# pad = (real.shape[-1] - test.shape[-1]) // 2
pad = 40

patch1 = torch.cuda.FloatTensor(real[:, :mid+pad, :mid+pad]).unsqueeze(0)
patch2 = torch.cuda.FloatTensor(real[:, mid-pad:, :mid+pad]).unsqueeze(0)
patch3 = torch.cuda.FloatTensor(real[:, :mid+pad, mid-pad:]).unsqueeze(0)
patch4 = torch.cuda.FloatTensor(real[:, mid-pad:, mid-pad:]).unsqueeze(0)

patches = [patch1, patch2, patch3, patch4]
fakes = []
for patch in patches:
    test = net(patch)
    fakes.append(test.detach().cpu().squeeze())

fake_comb = torch.cat((torch.cat((fakes[0][:-pad, :-pad], fakes[1][pad:, :-pad])), torch.cat((fakes[2][:-pad, pad:], fakes[3][pad:, pad:]))), axis=1)
# fake_comb = torch.cat((torch.cat((fakes[0], fakes[1])), torch.cat((fakes[2], fakes[3]))), axis=1)


# %%
plt.figure(figsize=(10,10))
plt.imshow(fake_comb, cmap='gray')#, vmin=fake.min(), vmax=fake.max())

# %% Weight analyses:
weights = cycleGun.netG1.model[20].weight.detach()

in_hists = []
for weight in weights:
    in_hists += [torch.histc(abs(weight))]
in_hists = torch.cat(in_hists).reshape((len(in_hists), len(in_hists[0])))

plt.figure(figsize=(10,10))
plt.imshow(in_hists.detach())
plt.colorbar()

#%% 
out_hists = []
for weight in weights.permute((1,0,2,3)):
    out_hists += [torch.histc(abs(weight))]
out_hists = torch.cat(out_hists).reshape((len(out_hists), len(out_hists[0])))

plt.figure(figsize=(10,10))
plt.imshow(out_hists.detach())
plt.colorbar()

#%%
request = gp.BatchRequest()
for array in [self.real_A, self.mask_A]:            
    extents = self.get_extents(512, array_name=array.identifier)
    request.add(array, self.common_voxel_size * extents, self.common_voxel_size)

# %%
datapipe = self.datapipe_A
predict_pipe = datapipe.source + gp.RandomLocation() 
if datapipe.reject: predict_pipe += datapipe.reject
if datapipe.resample: predict_pipe += datapipe.resample
predict_pipe += datapipe.normalize_real
predict_pipe += datapipe.scaleimg2tanh_real
if datapipe.unsqueeze: # add "channel" dimensions if neccessary, else use z dimension as channel
    predict_pipe += datapipe.unsqueeze
predict_pipe += gp.Unsqueeze([datapipe.real]) # add batch dimension

with gp.build(predict_pipe):
    in_batch = predict_pipe.request_batch(request)

# %%
with gp.build(self.pipe_A):
    in_batch = self.pipe_A.request_batch(request)

# %%
plt.figure(figsize=(10,10))
plt.imshow(in_batch[cycleGun.real_A].data.squeeze(), cmap='gray')
# %%
# outs = cycleGun.model(torch.cuda.FloatTensor(in_batch[cycleGun.real_A].data))
outs = cycleGun.model(torch.FloatTensor(in_batch[cycleGun.real_A].data))
fake2 = outs[0].detach().cpu().squeeze()
# %%
plt.figure(figsize=(10,10))
plt.imshow(fake2, cmap='gray', vmin=fake2.min(), vmax=fake2.max())

# %%
plt.figure(figsize=(10,10))
plt.imshow(test_batch[cycleGun.fake_B].data.squeeze(), cmap='gray')
# %%
cycleGun._get_latest_checkpoint()
cycleGun.load_saved_model()

# %%
import zarr
import numpy as np
import matplotlib.pyplot as plt
import daisy

# %%
z = zarr.open(cycleGun.src_B)
shape = np.array(z[cycleGun.B_name].shape)
mid = shape // 2
im_data = np.array(z[cycleGun.B_name][mid[0], mid[1]-512:mid[1]+512, mid[2]-512:mid[2]+512]).squeeze()
# im = np.array(z['volumes/raw'][1400:1800, 1400:1800, 1400:1800])[200].squeeze()
z[cycleGun.B_name].info
plt.imshow(im_data, cmap='gray')


# %%
ds = daisy.open_ds(cycleGun.src_B, cycleGun.B_name)
roi = daisy.Roi((0, 0, 0), (40, 2048, 2048))
img = ds.to_ndarray(roi)

# %%
plt.figure(figsize=(30,30))
plt.imshow(img[0].squeeze(), cmap='gray')

# %%
z = zarr.open('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_bottomp100um_30nm_rectwopassdb9_.n5')
# im_data = np.array(z['volumes/raw'][400:600, 10000:14000, 10000:14000]).squeeze()
# im = np.array(z['volumes/raw'][1400:1800, 1400:1800, 1400:1800])[200].squeeze()
z['volumes']['raw'].info
# plt.imshow(im_data[55], cmap='gray')

# %%
ds = daisy.open_ds('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_bottomp100um_30nm_rectwopassdb9_.n5', 'volumes/raw')
roi = daisy.Roi((48240, 0, 0), (30, 96480, 96480))
img = ds.to_ndarray(roi)

# %%
plt.figure(figsize=(30,30))
plt.imshow(img[0].squeeze(), cmap='gray')

# %%
ds = daisy.open_ds('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/cb2_gcl1.n5', 'volumes/raw')
# ds = daisy.open_ds('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_bottomp100um_30nm_rectwopassdb9_.n5', 'volumes/raw')
roi = daisy.Roi((2000+22000, 520000+46000, 744000+46000), (40, 8000, 8000))
img = ds.to_ndarray(roi)

# %%
plt.figure(figsize=(30,30))
plt.imshow(img[0].squeeze(), cmap='gray')

# %%
z = zarr.open('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBm_FN_lobX_90nm_tile2_rec_db9_twopass_full_.n5')
y = zarr.open('/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/CBm_FN_lobX_90nm_tile2_rec_db9_twopass_full_.n5')

full = np.array(z['volumes']['raw'][1400:1800, 1400:1800, 1400:1800]).squeeze()
quart = np.array(z['volumes']['quarterAngle'][1400:1800, 1400:1800, 1400:1800]).squeeze()
# quart = np.array(z['volumes']['quarterAngle'][1400:1800, 1400:1800, 1400:1800]).squeeze()
# z['volumes']['quarterAngle'].info
# y['volumes']['raw'].info

fig, ax = plt.subplots(1, 2, figsize=(20,40))
ax[0].imshow(full[200,...], cmap='gray')
ax[1].imshow(quart[200,...], cmap='gray')

# %%
(quart == full).all()

# %%
z = zarr.open('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBm_FN_lobX_90nm_tile3_twopass_rec_.n5')
y = zarr.open('/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/CBm_FN_lobX_90nm_tile3_twopass_rec_.n5')

full = np.array(z['volumes']['raw'][1400:1800, 1400:1800, 1400:1800]).squeeze()
quart = np.array(z['volumes']['quarterAngle'][1400:1800, 1400:1800, 1400:1800]).squeeze()
# quart = np.array(z['volumes']['quarterAngle'][1400:1800, 1400:1800, 1400:1800]).squeeze()
# z['volumes']['quarterAngle'].info
# y['volumes']['raw'].info

fig, ax = plt.subplots(1, 2, figsize=(20,40))
ax[0].imshow(full[200,...], cmap='gray')
ax[1].imshow(quart[200,...], cmap='gray')

# %%
z['volumes']['quarterAngle'].attrs.keys()

# %%
z = zarr.open('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobX_bottomp100um_30nm_rec_db9_.n5')
im = np.array(z['volumes']['raw'][1400:1800, 1400:1800, 1024]).squeeze()
z['volumes']['raw'].info

# %%
plt.figure(figsize=(20,20))
plt.imshow(im, cmap='gray')

# %% [markdown]
# # VALIDATION:

# %%
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from CycleGAN import *

working_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/'
src_A_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBm_FN_lobX_90nm_tile2_rec_db9_twopass_full_.n5'
src_B_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobX_bottomp100um_30nm_rec_db9_.n5'
model_name = 'CycleGun_CBxFN90nmtile3_CBx30nmBottom100um_20210930'

print('Loading model...')
cycleGun = CycleGAN(src_A=src_A_path, #EXPECTS ZARR VOLUME
                src_B=src_B_path,
                voxel_size=gp.Coordinate((30, 30, 30)), #voxel_size of B
                AB_voxel_ratio=3, #determines whether to add up/downsampling node (always rendered to B's voxel_size), (int or tuple of ints)
                A_name='volumes/raw',
                B_name='volumes/raw',
                # mask_A_name='volumes/masks/train_mask_20210925', # expects mask to be in same place as real zarr
                # mask_B_name='volumes/masks/train_mask_20210925',
                g_init_learning_rate=0.0004,
                d_init_learning_rate=0.0004,
                model_name = model_name,
                model_path = working_path+'models/',
                tensorboard_path = working_path+'tensorboard/'+model_name,
                num_epochs = 20000,
                )                

# %%
cycleGun.set_device(3)

# %%
cycleGun.test_train()

# %%
cycleGun.test_prediction(in_type='A', side_length=80)

# %%
cycleGun.test_prediction(in_type='B', side_length=80)


