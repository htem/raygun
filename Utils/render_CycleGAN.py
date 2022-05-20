import argparse
import sys
import runpy
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from CycleGAN import *

def render_tiled(script_path, 
                side, 
                src_path, 
                src_name, 
                checkpoints, 
                side_length=64, 
                cycle=False, 
                crop_to_valid="false",
                num_workers=30, 
                ):
    
    #Load Model
    script_dict = runpy.run_path(script_path)
    cycleGun = script_dict['cycleGun']

    #Set Dataset to Render
    ds = daisy.open_ds(src_path, src_name)
    src_voxel_size = ds.voxel_size
    side = side.upper()
    if type(checkpoints) is not list:
        checkpoints = [checkpoints]

    setattr(cycleGun, f'src_{side}', src_path)
    setattr(cycleGun, f'{side}_out_path', src_path)
    setattr(cycleGun, f'{side}_name', src_name)
    setattr(cycleGun, f'{side}_voxel_size', src_voxel_size)
    setattr(cycleGun, f'mask_{side}_name', None)
    cycleGun.num_workers = num_workers

    cycleGun.build_machine()

    if len(cycleGun.model_name.split('-')) > 1:
        parts = cycleGun.model_name.split('-')
        label_prefix = parts[0] + parts[1][1:]
    else:
        label_prefix = cycleGun.model_name
    label_prefix += f'_seed{script_dict["seed"]}'

    if crop_to_valid.lower() == 'true':
        crop_to_valid = True
    elif crop_to_valid.lower() == 'false':
        crop_to_valid = False
    else:
        crop_to_valid = int(crop_to_valid)
    
    net = 'netG1' if side == 'A' else 'netG2'
    other_net = 'netG1' if side == 'B' else 'netG2'

    for checkpoint in checkpoints:
        this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{checkpoint}'
        cycleGun.checkpoint = this_checkpoint
        # cycleGun.load_saved_model(this_checkpoint)
        label_dict = {}
        label_dict['fake'] = f'{label_prefix}_checkpoint{checkpoint}_{net}'
        label_dict['cycled'] = f'{label_prefix}_checkpoint{checkpoint}_{net}{other_net}'
        
        print(f"Rendering checkpoint {this_checkpoint}...")
        cycleGun.render_full(side_length=int(side_length), side=side, cycle=cycle, crop_to_valid=crop_to_valid, label_dict=label_dict)
        print(f"Rendered!")
    
    print('Done.')


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("script_path", type=str, help='Path to CycleGAN training script.')
    ap.add_argument("side", type=str, help='Which side of the CycleGAN to render from (e.g. "A" or "B")')
    ap.add_argument("src_path", type=str, help='Input path to Zarr volume to render through network(s).')
    ap.add_argument("src_name", type=str, help='Input name of dataset in the Zarr volume to render through network(s).')
    ap.add_argument("checkpoints", type=int, help='Which training checkpoint(s) to render (eg. [300000, 325000, 350000]).', nargs='+')
    ap.add_argument("--side_length", type=int, help='Side length of volumes for rendering (in common voxels).', default=64)
    ap.add_argument("--cycle", type=bool, help='Whether or not to render both network passes.', default=False)
    ap.add_argument("--crop_to_valid", type=str, help='Whether to crop network outputs and, optionally, by how much (e.g. "false" or "true" or "32").', default='false')
    ap.add_argument("--num_workers", type=int, help='How many workers to run in parallel.', default=30)
    config = ap.parse_args()
    
    render_tiled(**vars(config))
    #TODO: ADD MINIMAL VERSION