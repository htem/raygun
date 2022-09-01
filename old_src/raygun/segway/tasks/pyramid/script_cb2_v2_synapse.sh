
f="/n/vast/htem/Segmentation/cb2_v3/output.zarr"
ds="volumes/pred_syn_indicator"
output_file='/n/pure/htem/Segmentation/cb2_v3/output.zarr'
roi_offset='3800 114688 131072'
roi_shape='42600 524288 524288'
# roi_shape='8000 8192 8192'

# mkdir -p ${output_file}/${ds}_mipmap
# ln -s `realpath ${f}/${ds}` ${output_file}/${ds}_mipmap/s0

# in_ds=${ds}_mipmap/s0
# out_ds=${ds}_mipmap/s1
# scale_factor='1 2 2' # from 40x16x16 to 40x32x32
# python segway/tasks/pyramid/task_scale_pyramid2.py $f $ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

# in_ds=${ds}_mipmap/s1
# out_ds=${ds}_mipmap/s2
# scale_factor='1 2 2' # from 40x32x32 to 40x64x64
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

# in_ds=${ds}_mipmap/s2
# out_ds=${ds}_mipmap/s3
# scale_factor='2 2 2' # to 80x128x128
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

# in_ds=${ds}_mipmap/s3
# out_ds=${ds}_mipmap/s4
# scale_factor='2 2 2' # to 160x256x256
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

in_ds=${ds}_mipmap/s4
out_ds=${ds}_mipmap/s5
scale_factor='2 2 2' # to 320x512x512
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

in_ds=${ds}_mipmap/s5
out_ds=${ds}_mipmap/s6
scale_factor='2 2 2' # to 640x1024x1024
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

in_ds=${ds}_mipmap/s6
out_ds=${ds}_mipmap/s7
scale_factor='2 2 2' # to 1280x2048x2048
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8

