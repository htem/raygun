#SCRIPT TO RECHUNK EXISTING ZARRs/N5s

#all in zyx

# write_size="1 256 256"
# python /n/groups/htem/users/jlr54/raygun/segway/tasks/rechunk/task_rechunk.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/cb2_gcl1.n5 volumes/raw /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/cb2_gcl1.n5 volumes/raw_sliced --write_size $write_size --roi_offset 2800 520000 744000 --roi_shape 44080 96480 96480 --num_workers 32


write_size="1 256 256"
python /n/groups/htem/users/jlr54/raygun/segway/tasks/rechunk/task_rechunk.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_bottomp100um_30nm_rec_db9_.n5 volumes/raw /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_bottomp100um_30nm_rec_db9_.n5 volumes/raw_sliced --write_size $write_size --roi_offset 0 0 0 --roi_shape 96480 96480 61440 --num_workers 32

