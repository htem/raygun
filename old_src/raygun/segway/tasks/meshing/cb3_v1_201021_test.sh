# python segway/tasks/meshing/task_meshing.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s3 /n/vast/htem/Segmentation/cb2_v3/output.zarr/meshes_s3/precomputed --block_size 4000 8192 8192 --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --no_launch_workers 1
# python segway/tasks/meshing/task_meshing.py /n/vast/htem/Segmentation/cb2_v3/output.zarr volumes/super_1x2x2_segmentation_0.400_mipmap/s0 /n/vast/htem/Segmentation/cb2_v3/output.zarr/meshes_new/precomputed --block_size 4000 8192 8192 --roi_offset 3800 114688 131072 --roi_shape 4000 16384 16384 --no_launch_workers 0 --downsample 2 4 4

            # "sub_roi_offset": [120, 262144, 262144],
            # "sub_roi_shape": [20000, 262144, 262144],

i=5
f="/n/groups/htem/Segmentation/cb3_201019.n5"
# ds="volumes/super_1x2x2_segmentation_0.500"
ds="volumes/super_1x2x2_segmentation_0.${i}00_mipmap/s2"  # 40x32x32
output_dir="/n/f810/htem/Segmentation/cb3_201019.n5/meshes/super_1x2x2_segmentation_0.${i}00_mipmap"
roi_offset='120 262144 262144'
roi_shape='20000 262144 262144'
block_size='5120 8192 8192'
downsample='1 2 2'  # to 40x64x64
max_quadrics_error=3e-28
min_obj_size=32  # for 40x64x64 voxels

python segway/tasks/meshing/task_meshing2_v2.py $f $ds ${output_dir} --block_size $block_size --roi_offset $roi_offset --roi_shape $roi_shape --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size --no_launch_workers 0 --downsample $downsample --num_workers 8 --overwrite 2
# python segway/tasks/meshing/task_meshing2_v2.py $f $ds ${output_dir} --block_size $block_size --roi_offset $roi_offset --roi_shape $roi_shape --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size --no_launch_workers 0 --downsample $downsample --num_workers 8 --overwrite 0 --config_hash 311492761269462973275375375227472116636
