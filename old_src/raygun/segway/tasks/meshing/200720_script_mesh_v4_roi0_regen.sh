# python segway/tasks/meshing/task_meshing.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s3 /n/vast/htem/Segmentation/cb2_v3/output.zarr/meshes_s3/precomputed --block_size 4000 8192 8192 --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --no_launch_workers 1
# python segway/tasks/meshing/task_meshing.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s0 /n/vast/htem/Segmentation/cb2_v3/output.zarr/meshes_new/precomputed --block_size 4000 8192 8192 --roi_offset 3800 114688 131072 --roi_shape 4000 16384 16384 --no_launch_workers 0 --downsample 2 4 4

f="/n/balin_tank_ssd1/htem/Segmentation/cb2_v4/output.zarr"
# ds="volumes/super_1x2x2_segmentation_0.500"
ds="volumes/super_1x2x2_segmentation_0.500_mipmap/s2"  # 40x32x32
output_file='/n/balin_tank_ssd1/htem/Segmentation/cb2_v4/output.zarr'
roi_offset='2800 114688 131072'  # roi0
roi_shape='44000 425984 786432'  # roi0
# roi_offset='2800 540672 131072'  # roi1
# roi_shape='44000 344064 786432'  # roi1
block_size='4000 8192 8192'
downsample='1 2 2'  # to 40x64x64
max_quadrics_error=1e8
min_obj_size=32  # for 40x64x64 voxels

# test
# roi_shape='4000 16384 16384'
python segway/tasks/meshing/task_meshing2.py $f $ds ${output_file}/meshes/precomputed --block_size $block_size --roi_offset $roi_offset --roi_shape $roi_shape --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size --no_launch_workers 1 --downsample $downsample --num_workers 8 --overwrite 0 --db_name MeshingTask_155041869855057631652572757205105972729
# 133MB

# test: max_quadric_error = 1e8 (from 1e6)
# 83MB

roi_shape='8000 16384 16384'
# 65M @ 1e6

downsample='2 8 8'
max_quadrics_error='1e8'
# 58M @ 1e8

downsample='1 8 8'
max_quadrics_error='1e8'
# 83MB

# min_obj_size=512
# 25M

# = 256
# 28MB

roi_shape='4000 16384 16384'
min_obj_size=64
# 19MB
min_obj_size=32
# 20MB

max_quadrics_error=1e8
# 16MB
max_quadrics_error=2e8

