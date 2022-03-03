fpath='/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBm_FN_lobX_90nm_tile3_twopass_rec_.n5'
output_ds='volumes/masks/train_mask_20210925'

python add_validation_mask.py $fpath --output_ds $output_ds