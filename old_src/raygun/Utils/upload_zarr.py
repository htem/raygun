#!/usr/bin/env python3
import sys
from cloudvolume import CloudVolume
import daisy

import downsample_GC

def uploadZarr2GC(path, ds_name, cloud_path):

	# --- Load image volume and upload --- #
	print('Loading volume into memory...')
	ds = daisy.open_ds(path, ds_name)
	
	# --- Parameters and paths --- #
	description = (
		"TODO: add description" + path
	)
	owners = [
		'joitapac@esrf.fr',
		'aaron.t.kuan@gmail.com',
		'wei-chung_lee@hms.harvard.edu'
	]

	# --- Create or connect to a cloudvolume --- #
	info = CloudVolume.create_new_info(
		num_channels = 1 if ds.n_channel_dims == 0 else ds.chunk_shape[0],
		layer_type = 'image', # 'image' or 'segmentation'
		data_type = ds.dtype,#'uint8', # can pick any popular uint
		encoding = 'raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
		resolution = ds.voxel_size, # X,Y,Z values in nanometers
		voxel_offset = ds.data_roi.get_offset() / ds.voxel_size, # values X,Y,Z values in voxels
		chunk_size = ds.chunk_shape[ds.n_channel_dims:], # rechunk of image X,Y,Z in voxels
		volume_size = ds.data.shape[ds.n_channel_dims:] #ds.shape, # X,Y,Z size in voxels
	)

	print(f'Opening a cloudvolume at {cloud_path}')
	vol = CloudVolume(cloud_path, info=info)

	def push_metadata():
		vol.provenance.description = description
		vol.provenance.owners = owners
		vol.commit_info() # generates gs://bucket/dataset/layer/info json file
		vol.commit_provenance() # generates gs://bucket/dataset/layer/provenance json

	push_metadata()  # Only needs to be done once per volume


	print('Uploading to google cloud...')
	if ds.n_channel_dims == 0:
		assert vol.shape[:3] == ds.shape, f'Dataset shape {ds.shape} does not match CloudVolume shape {vol.shape}'
		vol[:,:,:,0] = ds.to_ndarray()
	else:
		assert vol.shape == ds.shape[1:] + tuple([ds.shape[0]]), f'Dataset shape {ds.shape[1:] + tuple([ds.shape[0]])} does not match CloudVolume shape {vol.shape}'
		vol[...] = ds.to_ndarray().transpose((1,2,3,0)) #TODO: Assumes 1 channel dimension
	print('Done.')

if __name__ == "__main__":
	uploadZarr2GC(sys.argv[1], sys.argv[2], sys.argv[3])

	if len(sys.argv) >= 5 and 'downsample' in sys.argv[4:]:
		print('Downsampling CloudVolume... You may want to "watch ls igneous_tasks/queue/" to see when all tasks are complete, and then kill this job manually.')
		downsample_GC.create_task_queue(sys.argv[3])
		downsample_GC.run_tasks_from_queue()
		print('Done.')
		