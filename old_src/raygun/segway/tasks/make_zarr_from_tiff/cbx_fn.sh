
# go to working directory
cd /n/groups/htem/Segmentation/shared-nondev/cbx_fn

prefix='CBm_FN_lobX_90nm_tile0_rec_db9_twopass'
aligned_dir_path="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}.n5"
output_dataset="volumes/raw"
voxel_size='90 90 90'
roi_offset='0 0 0'
roi_shape='289440 289440 289440'
x_tile_size=3216
y_tile_size=3216
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4


prefix='CBm_FN_lobX_90nm_tile1_rec_db9_twopass'
aligned_dir_path="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}.n5"
output_dataset="volumes/raw"
voxel_size='90 90 90'
roi_offset='0 0 0'
roi_shape='289440 289440 289440'
x_tile_size=3216
y_tile_size=3216
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4


prefix='CBm_FN_lobX_90nm_tile2_rec_db9_twopass_full'
aligned_dir_path="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}.n5"
output_dataset="volumes/raw"
voxel_size='90 90 90'
roi_offset='0 0 0'
roi_shape='289440 289440 289440'
x_tile_size=3216
y_tile_size=3216
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4


prefix='CBm_FN_lobX_90nm_tile3_twopass_rec'
aligned_dir_path="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}.n5"
output_dataset="volumes/raw"
voxel_size='90 90 90'
roi_offset='0 0 0'
roi_shape='289440 289440 289440'
x_tile_size=3216
y_tile_size=3216
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4


prefix='CBm_FN_lobX_90nm_tile4_rec_db9_twopass'
aligned_dir_path="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}.n5"
output_dataset="volumes/raw"
voxel_size='90 90 90'
roi_offset='0 0 0'
roi_shape='289440 289440 289440'
x_tile_size=3216
y_tile_size=3216
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4


prefix='CBm_FN_lobX_90nm_tile5_rec_db9_twopass'
aligned_dir_path="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/volraw/${prefix}.n5"
output_dataset="volumes/raw"
voxel_size='90 90 90'
roi_offset='0 0 0'
roi_shape='289440 289440 289440'
x_tile_size=3216
y_tile_size=3216
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4

