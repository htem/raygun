

# write_size="16 128 128"
# # 16x128x128 = 256k voxels, optimized for neuroglancer with max_voxel_log2=18
# # for voxel size 40x8x8
# python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s1 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s1_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 32

write_size="32 64 128"
# optimized for voxel size 40x16x16
# 32x64x64 = 128k
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s1 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s1_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 32


write_size="64 64 64"
# optimized for voxel size 40x32x32
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s2 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s2_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 32

write_size="64 64 64"
# optimized for almost isotropic voxel sizes
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s3 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s3_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 16

write_size="64 64 64"
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s4 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s4_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 16

write_size="64 64 64"
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s5 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s5_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 16

write_size="64 64 64"
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s6 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s6_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 16

write_size="64 64 64"
python segway/tasks/rechunk/task_rechunk.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s7 /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s7_rechunked --write_size $write_size --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 16

