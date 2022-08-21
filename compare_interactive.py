# %%
# !conda activate n2v
%load_ext autoreload
# from noise2gun import *
from compare import *

# %% [markdown]
# # Define and Load datasets
# Must be ZARRs

# %% [markdown]
# ## Define

# %%
# src_path = src = "/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/jlr54_tests/volumes/mCTX_17keV_30nm_512c_first256/mCTX_17keV_30nm_512c_first256.zarr/volumes"
# gt_name = 'raw'
# ds_names = [NOT CYCLEGAN]
src_path = src = "/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5"
ds_names=['volumes/raw_30nm','volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed42_checkpoint340000_netG2_184tCrp']

# %% [markdown]
# ## Load

# %%
comparator = Compare(src_path, # 'path/to/data.zarr/volumes'
                # gt_name=gt_name, # 'gt_dataset_name'
                ds_names=ds_names, # ['dataset_1_name', 'dataset_2_name', ...]
                out_path=None,
                batch_size=1,
                metric_list=None,
                vizualize=True,
                # make_mask=True
    )

# %%
batch = comparator.patch_compare()

# %%
comparator.batch_show()

# %%
results = comparator.compare()

# %%
norm_results = results.divide(results.train, axis='rows').drop(['compare_mask', 'train'], axis=1) - 1
comparator.plot_results(norm_results)

# %%
import zarr
import daisy

# %%
n2g = Noise2Gun('', gp.Coordinate((30,30,30)), side_length=34)
pad = (n2g.context_side_length - n2g.side_length) // 2

for array, specs in comparator.source.array_specs.items():
    ds = comparator.source.datasets[array]
    d = daisy.open_ds(src_path.rstrip('/volumes'), 'volumes/'+ds)
    crop = gp.Coordinate(pad * d.voxel_size)
    spec = specs
    roi = gp.Roi(d.roi.get_offset(), d.roi.get_shape())
    spec.roi = roi.grow(-crop, -crop)
    spec.voxel_size = d.voxel_size
    spec.dtype = d.dtype
    comparator.source.array_specs[array] = spec

# %%
for array, specs in comparator.source.array_specs.items():
    print(array)
    print(specs.roi)

# %%
# comparator.ds_names.append(comparator.mask_name)
comparator.make_pipes()

# %%
comparator.source.datasets

# %%
z = zarr.open(src_path)
z['compare_mask'].info
# z['gt'].info

# %%



