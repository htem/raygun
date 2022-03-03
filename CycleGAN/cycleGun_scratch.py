# %%
import sys
sys.path.append('/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/')
from CycleGun_CBv30nmBottom100um_cb2gcl1_20220215_ import *
import matplotlib.pyplot as plt
import zarr


# %%
# cycleGun.set_device(0)
# cycleGun.load_saved_model()
cycleGun.batch_size = 1

# %%
batch = cycleGun.test_train()

# %%
pred_batch = cycleGun.test_prediction('B', side_length=512, cycle=True)

# %%
test = cycleGun.model(torch.cuda.FloatTensor(pred_batch[cycleGun.real_A].data).unsqueeze(0))
plt.figure(figsize=(20,20))
plt.imshow(test[0].squeeze().detach().cpu(), cmap='gray')

# %%
cycleGun._get_latest_checkpoint()
cycleGun.load_saved_model()

# %%
import zarr
import numpy as np
import matplotlib.pyplot as plt
import daisy

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

