#!/usr/bin/env python3
import numpy as np
from cloudvolume import CloudVolume
import daisy

def loadVol_uploadGC(volume_fn,cloud_path,volume_size,voxel_size):
	# --- Parameters and paths --- #
	#volume_size = [3216, 3216, 2048]  # In voxels, xyz order
	#voxel_size = 90  # In nm

	
	#volume_fn = '/home/esrf/ls2892/id16a/ctx_paraffin/volraw/ctx_paraffin_tile02_090nm_rec_db9_.raw'
	description = (
		"TODO: add description" + volume_fn
	)
	owners = [
		'joitapac@esrf.fr',
		'aaron.t.kuan@gmail.com',
		'wei-chung_lee@hms.harvard.edu'
	]


	# --- Create or connect to a cloudvolume --- #
	info = CloudVolume.create_new_info(
		num_channels = 1,
		layer_type = 'image', # 'image' or 'segmentation'
		data_type = 'uint8', # can pick any popular uint
		encoding = 'raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
		resolution = [voxel_size, voxel_size, voxel_size], # X,Y,Z values in nanometers
		voxel_offset = [0, 0, 0], # values X,Y,Z values in voxels
		chunk_size = [64, 64, 64], # rechunk of image X,Y,Z in voxels
		volume_size = volume_size, # X,Y,Z size in voxels
	)

	print(f'Opening a cloudvolume at {cloud_path}')
	vol = CloudVolume(cloud_path, info=info)

	def push_metadata():
		vol.provenance.description = description
		vol.provenance.owners = owners
		vol.commit_info() # generates gs://bucket/dataset/layer/info json file
		vol.commit_provenance() # generates gs://bucket/dataset/layer/provenance json

	push_metadata()  # Only needs to be done once per volume


	# --- Load image volume and upload --- #
	print('Loading volume into memory...')
	im = daisy.open_ds(volume_fn, dtype=np.uint8)
	# print('Reshaping...')
	# im = im.reshape(*volume_size, order='F')

	print('Uploading to google cloud...')
	assert vol.shape[:3] == im.shape
	vol[:, :, :] = im
	print('Done.')