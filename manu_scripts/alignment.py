# %%
import matplotlib.pyplot as plt
import daisy
import SimpleITK as sitk

# %%
moving_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0_unaligned90nm.n5'
moving_name = 'volumes/raw_90nm'
moving_ds = daisy.open_ds(moving_file, moving_name)
moving_array = moving_ds.to_ndarray()

fixed_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
fixed_name = 'volumes/raw_30nm'
fixed_ds = daisy.open_ds(fixed_file, fixed_name)
fixed_array = fixed_ds.to_ndarray()

destination_name ='volumes/interpolated_90nm_aligned'

# %%
elx = sitk.ElastixImageFilter()
fixed_im = sitk.GetImageFromArray(fixed_array)
fixed_im.SetSpacing((0.03, 0.03, 0.03))
moving_im = sitk.GetImageFromArray(moving_array)
moving_im.SetSpacing((0.09, 0.09, 0.09))

# %%
elx.SetFixedImage(fixed_im)
elx.SetMovingImage(moving_im)
elx.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
elx.SetParameter('WriteResultImage', 'false')
elx.SetParameter('AutomaticScalesEstimation', 'false')
elx.SetParameter('CheckNumberOfSamples', 'false')
elx.Execute()
aligned_im = elx.GetResultImage()
aligned_array = sitk.GetArrayFromImage(aligned_im)
aligned_array = aligned_array.astype(fixed_array.dtype)

# %%
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
axs[0].imshow(moving_array[moving_array.shape[0]//2,...], cmap='gray')
axs[0].set_title('90nm unaligned')
axs[1].imshow(fixed_array[fixed_array.shape[0]//2,...], cmap='gray')
axs[1].set_title('30nm')
axs[2].imshow(aligned_array[aligned_array.shape[0]//2,...], cmap='gray')
axs[2].set_title('90nm aligned')
# %%
#Write result
source = daisy.Array(aligned_array, fixed_ds.roi, fixed_ds.voxel_size)

chunk_size = 64
num_channels = 1
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }
num_workers = 30
total_roi = daisy.Roi(fixed_ds.roi.get_offset(), fixed_ds.roi.get_shape())
write_size = fixed_ds.voxel_size * chunk_size
chunk_roi = daisy.Roi([0,]*len(fixed_ds.roi.get_offset()), write_size)

destination = daisy.prepare_ds(
    fixed_file, 
    destination_name,
    total_roi,
    fixed_ds.voxel_size,
    aligned_array.dtype,
    write_size=write_size,
    write_roi=chunk_roi,
    num_channels=num_channels,
    compressor=compressor)

#Prepare saving function/variables
def save_chunk(block:daisy.Roi):
    try:
        destination.__setitem__(block.write_roi, source.__getitem__(block.read_roi))
        return 0 # success
    except:
        return 1 # error
        
#Write data to new dataset
success = daisy.run_blockwise(
            total_roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit='shrink',
            num_workers=num_workers,
            max_retries=2)

if success:
    print(f'{total_roi} from {moving_file}/{moving_name} aligned and written to {fixed_file}/{destination_name}')
else:
    print('Failed to save cutout.')



# %%
