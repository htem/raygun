
f="/n/vast/htem/Segmentation/shared-nondev/gintylab-skin0/gt-skin0/cutout0/cutout0.zarr"
ds="volumes/raw_16_4"
output_file=$f
# roi_offset='2800 114688 131072'
# roi_shape='44000 425984 786432'
# roi_shape='44000 65536 65536'

# mkdir -p ${output_file}/${ds}_mipmap
# ln -s `realpath ${f}/${ds}` ${output_file}/${ds}_mipmap/s0

in_ds=volumes/raw_16_4
out_ds=volumes/raw_16
scale_factor='4 1 1' # to 16x16x16
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 16 --overwrite 0 --scale_factor ${scale_factor}

in_ds=volumes/seg_16_4
out_ds=volumes/seg_16
scale_factor='4 1 1' # to 16x16x16
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 16 --overwrite 0 --scale_factor ${scale_factor}

# in_ds=${ds}_mipmap/s0
# out_ds=${ds}_mipmap/s1
# scale_factor='1 2 2' # to 40x16x16
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 16 --overwrite 0 --scale_factor ${scale_factor} --scheduling_block_size_mult 4 4 4

# in_ds=${ds}_mipmap/s1
# out_ds=${ds}_mipmap/s2
# scale_factor='1 2 2' # to 40x32x32
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 16 --overwrite 0 --scale_factor ${scale_factor} --scheduling_block_size_mult 2 2 2

# in_ds=${ds}_mipmap/s2
# out_ds=${ds}_mipmap/s3
# scale_factor='2 2 2' # to 80x64x64
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 16 --overwrite 0 --scale_factor ${scale_factor} --scheduling_block_size_mult 2 2 2

# in_ds=${ds}_mipmap/s3
# out_ds=${ds}_mipmap/s4
# scale_factor='2 2 2' # to 160x128x128
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 16 --overwrite 0 --scale_factor ${scale_factor}

# in_ds=${ds}_mipmap/s4
# out_ds=${ds}_mipmap/s5
# scale_factor='2 2 2' # to 320x256x256
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 16 --overwrite 0 --scale_factor ${scale_factor}

# in_ds=${ds}_mipmap/s5
# out_ds=${ds}_mipmap/s6
# scale_factor='2 2 2' # to 640x512x512
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8 --overwrite 0 --scale_factor ${scale_factor}

# in_ds=${ds}_mipmap/s6
# out_ds=${ds}_mipmap/s7
# scale_factor='2 2 2' # to 1280x1024x1024
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8 --overwrite 0 --scale_factor ${scale_factor}

# in_ds=${ds}_mipmap/s7
# out_ds=${ds}_mipmap/s8
# scale_factor='2 2 2' # to 2560x2048x2048
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --roi_offset $roi_offset --roi_shape $roi_shape --num_workers 8 --overwrite 0 --scale_factor ${scale_factor}

