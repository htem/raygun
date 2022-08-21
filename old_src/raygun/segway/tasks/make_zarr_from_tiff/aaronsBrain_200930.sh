
aligned_dir_path="/n/groups/htem/temcagt/datasets/aaronsBrain/intersection/V1_aligned_tifs"
output_file='/n/groups/htem/temcagt/datasets/aaronsBrain/intersection/V1_aligned.n5'
output_dataset="volumes/raw"
voxel_size='45 4 4'
roi_offset='0 0 0'
### volume size calculation
# Z: 0-249 = 250*45 = 11250
# Y: r1-53 = 53*4*2048 = 434176
# X: c1-54 = 54*4*2048 = 442368
roi_shape='11250 434176 442368'
roi_offset='0 0 0'
y_tile_size=2048
x_tile_size=2048
section_dir_name_format="section_{:04d}"
bad_sections='180 202'

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --no_launch_workers 1 --bad_sections $bad_sections

