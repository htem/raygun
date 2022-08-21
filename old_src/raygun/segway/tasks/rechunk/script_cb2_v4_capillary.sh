
# f="/n/f810/htem/Segmentation/cb2_v4_test_roi0/output.zarr"
# ds="volumes/capillary_pred"
# output_file=$f
# python segway/tasks/rechunk/task_rechunk.py $f $ds ${output_file} ${ds}_rechunked --num_workers 16

f="/n/f810/htem/Segmentation/cb2_v4_test_roi0/output.zarr"
ds="volumes/capillary_pred_v7"
output_file=$f
python segway/tasks/rechunk/task_rechunk.py $f $ds ${output_file} ${ds}_rechunked --num_workers 16
