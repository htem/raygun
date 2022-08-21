
f="/n/vast/htem/Segmentation/cb2_v3/output.zarr"
ds="volumes/affs_mipmap"
for i in 7 6 5 4 3 2 1; do
    python segway/tasks/rechunk/task_rechunk_test.py $f $ds/s$i $f $ds/s${i}_rechunked --roi_offset 3800 114688 131072 --roi_shape 42600 524288 524288 --num_workers 16
done
