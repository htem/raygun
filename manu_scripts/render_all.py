#%%
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
from render_CycleGAN import render_tiled

# import logging
# logging.basicConfig(level=logging.DEBUG)

#%%
if __name__ == '__main__':
    # side = 'B'
    # src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
    # src_name = 'volumes/raw_30nm'

    side = 'A'
    src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_0.n5'
    src_name = 'volumes/interpolated_90nm_aligned'

    crop = 16
    total_roi_crop=312

    checkpoints = [
                310000, 
                320000, 
                330000, 
                340000, 
                350000
                ]
    script_base_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/'
    scripts = [
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train.py',
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_train.py',
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2SplitNoBottle_train.py',
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2LinkNoBottle_train.py',
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-3SplitNoBottle_train.py',
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-3LinkNoBottle_train.py',    
    ]

    for script in scripts:
        render_tiled(script_base_path+script,
                side,
                src_path,
                src_name,
                checkpoints,
                crop=crop,
                num_workers=34,
                total_roi_crop=total_roi_crop)
