
aligned_dir_path="/n/groups/htem/data/qz/200121_B2_final"
output_file='/n/groups/htem/data/qz/200121_B2_final.n5'
output_dataset="volumes/raw"
voxel_size='6 6 6'
roi_offset='0 0 0'
### volume size calculation
# Z: 13892*6 = 83352
# Y: 17500*6 = 105000
# X: 15068*6 = 90408
roi_shape='83352 105000 90408'
roi_offset='0 0 0'
y_tile_size=17500
x_tile_size=15068
section_dir_name_format="left_resliced{:05d}.tif"
# bad_sections='180 202'
max_voxel_count=1048576

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 1

