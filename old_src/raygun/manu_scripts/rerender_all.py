#%%
import json
import os
from distutils.dir_util import mkpath

import logging
logging.basicConfig(level=logging.INFO)

sbatch_base = [
    "#!/bin/bash \n",
    "#SBATCH -p gpu_connectomics \n",
    "#SBATCH --gres=gpu:1 \n",
    "#SBATCH --account=connectomics_contrib \n",
    "#SBATCH -t 12:00:00  \n",
    "#SBATCH -c 11 \n",
    "#SBATCH --mem=50GB \n",
    "module load gcc/9.2.0 \n",
    "module load cuda/11.2 \n",
]


#%%
if __name__ == '__main__':
    side = 'B'
    # src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
    src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_0.n5'
    src_name = 'volumes/raw_30nm'

    # side = 'A'
    # src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_1.n5'
    # src_name = 'volumes/interpolated_90nm_aligned'

    src_path_name = src_path.split('/')[-1][:-3]
    crop = 16
    total_roi_crop=184

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

    checkpoint_dict = {
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train.py': 340000,
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_train.py': 310000,
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2SplitNoBottle_train.py': 350000,
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2LinkNoBottle_train.py': 330000,
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-3SplitNoBottle_train.py': 340000,
        'CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-3LinkNoBottle_train.py': 310000
        }


    default_dict={
        'script_path':  '',
        'side':  side,
        'src_path':  src_path,
        'src_name':  src_name,
        'checkpoints':  [],
        'crop': crop,
        'total_roi_crop': total_roi_crop,
        'num_workers': 34, 
        # 'pre_rendered_suffix': '',
        'pre_rendered_suffix': None,
        'ds_suffix': f'_{total_roi_crop}tCrp'
    }

    mkpath(f'{src_path_name}/{src_name.split("/")[-1]}/side{side}/crop{crop}/tCrp{total_roi_crop}')
    os.chdir(f'{src_path_name}/{src_name.split("/")[-1]}/side{side}/crop{crop}/tCrp{total_roi_crop}')
    for script in scripts:
        mkpath(script.replace('_train.py', ''))
        os.chdir(script.replace('_train.py', ''))
        # for checkpoint in checkpoints:
        checkpoint = checkpoint_dict[script]
        if os.path.exists(f'{os.getcwd()}/c{checkpoint}.out'):
            with open(f'{os.getcwd()}/c{checkpoint}.out', 'r') as f:
                if 'Done.' in f.readlines()[-1]:
                    continue
        kwargs = default_dict.copy()
        kwargs['script_path'] = script_base_path+script
        kwargs['checkpoints'] = [checkpoint]
        with open(f'c{checkpoint}.json', "w") as config_file:
            json.dump(kwargs, config_file)
        with open(f'c{checkpoint}.sbatch', 'w') as sbatch_file:
            sbatch_file.writelines(sbatch_base[:-2] + 
                    [f"#SBATCH -o c{checkpoint}.out \n"] +
                    sbatch_base[-2:] +
                    [f'python /n/groups/htem/users/jlr54/raygun/manu_scripts/render.py {os.getcwd()}/c{checkpoint}.json \n'])
        print(f'Starting {script} - {checkpoint}') 
        os.system(f'sbatch c{checkpoint}.sbatch')
        #__end for
        os.chdir('../')