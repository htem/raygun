# %%
import zarr
import numpy as np
import matplotlib.pyplot as plt
import daisy
import os
os.environ["ELASTIX_PATH"] = '/n/groups/htem/users/jlr54/elastix-5.0.1-linux'
import pyelastix as xx

# %%
moving_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0_unaligned90nm.n5'
moving_name = 'volumes/raw_90nm'
moving_ds = daisy.open_ds(moving_file, moving_name)
moving_array = moving_ds.to_ndarray()

fixed_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
fixed_name = 'volumes/raw_30nm'
fixed_ds = daisy.open_ds(fixed_file, fixed_name)
fixed_array = fixed_ds.to_ndarray()

# %%
params = xx.get_advanced_params()
params += xx.get_default_params()
params.FixedInternalImagePixelType = fixed_array.dtype
params.MovingInternalImagePixelType = moving_array.dtype
params.UseDirectionCosines = True
params.Registration = 'MultiResolutionRegistration'
params.FixedImagePyramid = 'FixedRecursiveImagePyramid'
params.MovingImagePyramid = 'MovingRecursiveImagePyramid'
params.HowToCombineTransforms = 'Compose'
params.DefaultPixelValue = 0
params.Interpolator = 'BSplineInterpolator'
params.BSplineInterpolationOrder = 1
params.ResampleInterpolator = 'FinalBSplineInterpolator'
params.FinalBSplineInterpolationOrder = 3
params.Resampler = 'DefaultResampler'
params.Metric = 'AdvancedMattesMutualInformation'
params.NumberOfHistogramBins = 32
params.ImageSampler = 'RandomCoordinate'
params.NumberOfSpatialSamples = 2048
params.NewSamplesEveryIteration = True
params.NumberOfResolutions = 4
params.Transform = 'BSplineTransform'
params.FinalGridSpacingInPhysicalUnits = 30
params.Optimizer = 'AdaptiveStochasticGradientDescent'
params.MaximumNumberOfIterations = 500

# %%
moving_aligned, field = xx.register(moving_array, fixed_array, params)

# %%
# Try with SimpleElastix:
import SimpleITK as sitk
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
elx.SetParameter('CheckNumberOfSamples', 'false')
elx.Execute()
aligned_im = elx.GetResultImage()
aligned_array = sitk.GetArrayFromImage(aligned_im)

# %%
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
axs[0].imshow(moving_array[moving_array.shape[0]//2,...], cmap='gray')
axs[0].set_title('90nm unaligned')
axs[1].imshow(fixed_array[fixed_array.shape[0]//2,...], cmap='gray')
axs[1].set_title('30nm')
axs[2].imshow(aligned_array[aligned_array.shape[0]//2,...], cmap='gray')
axs[2].set_title('90nm aligned')
# %%
