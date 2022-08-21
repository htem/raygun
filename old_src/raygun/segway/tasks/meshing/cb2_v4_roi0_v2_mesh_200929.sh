# python segway/tasks/meshing/task_meshing.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s3 /n/vast/htem/Segmentation/cb2_v3/output.zarr/meshes_s3/precomputed --block_size 4000 8192 8192 --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --no_launch_workers 1
# python segway/tasks/meshing/task_meshing.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s0 /n/vast/htem/Segmentation/cb2_v3/output.zarr/meshes_new/precomputed --block_size 4000 8192 8192 --roi_offset 3800 114688 131072 --roi_shape 4000 16384 16384 --no_launch_workers 0 --downsample 2 4 4

f="/n/f810/htem/Segmentation/cb2_v4/output.zarr"
# ds="volumes/super_1x2x2_segmentation_0.500"
ds="volumes/super_1x2x2_segmentation_0.500_mipmap/s2"  # 40x32x32
output_file='/n/f810/htem/Segmentation/cb2_v4/output.zarr'
roi_offset='2800 114688 131072'
roi_shape='44000 425984 786432'
# roi_shape='44000 16384 16384'
block_size='4000 8192 8192'
downsample='1 2 2'  # to 40x64x64
# max_quadrics_error=1e8  # for nm
# max_quadrics_error=1e8  # for nm
# max_quadrics_error=1e5
# max_quadrics_error=1e154
max_quadrics_error=3e-28
min_obj_size=32  # for 40x64x64 voxels

# 2.6e10
# 1e54

# test
# roi_shape='4000 16384 16384'
python segway/tasks/meshing/task_meshing2_v2.py $f $ds ${output_file}/meshes/precomputed_v2 --block_size $block_size --roi_offset $roi_offset --roi_shape $roi_shape --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size --no_launch_workers 0 --downsample $downsample --num_workers 8 --overwrite 2
# 133MB

/n/f810/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed/mesh/2/6628/6977/324
max_quadrics_error=1e5  # 5.5K
max_quadrics_error=1e8  # 5.5K
old: 30K


/n/f810/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed_v2/mesh/2/6962/8199/636
1e5: 65k
old: 285k

/n/f810/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed_v2/mesh/2/2824/8244/3269
3e15: 34k
old: 131752
max_quadrics_error=1e-27: 64k
max_quadrics_error=5e-28: 79624
max_quadrics_error=3e-28: 93k
max_quadrics_error=1e-28: 130k
max_quadrics_error=1e-30: 592948


