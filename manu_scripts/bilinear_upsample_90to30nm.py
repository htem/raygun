from gunpowder import gunpowder as gp

#CONFIG
src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_overview_90nm_rec5iter_db9_l20p15_.n5'
src_name = 'volumes/raw'
src_voxel_size=gp.Coordinate((90, 90, 90)), #voxel_size of source

dst_name = 'volumes/bilinear_30nm'
dst_voxel_size=gp.Coordinate((30, 30, 30)), #voxel size to cast all data into


#MAKE PIPELINE
src_key = gp.ArrayKey('SRC')
out = gp.ArrayKey('OUT')
resample = gp.Resample(src_key, dst_voxel_size, out, interp_order=1)



#RUN