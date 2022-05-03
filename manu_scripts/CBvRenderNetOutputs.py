import sys
import runpy
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from CycleGAN import *

if __name__ == '__main__':
    # python CBvRenderNetOutputs.py <Path to Training Script> <Source Path> <Source Name> <cycle> <side_length> <checkpoint1> ... <checkpointN>
    
    #Load Model
    script_dict = runpy.run_path(sys.argv[1])
    cycleGun = script_dict['cycleGun']

    #Set Dataset to Render
    cycleGun.src_A = sys.argv[2]
    cycleGun.A_out_path = sys.argv[2]
    cycleGun.A_name = sys.argv[3]
    cycleGun.mask_A_name = None
    cycleGun.build_machine()

    if len(cycleGun.model_name.split('-')) > 1:
        parts = cycleGun.model_name.split('-')
        label_prefix = parts[0] + parts[1][1:]
    else:
        label_prefix = cycleGun.model_name
    label_prefix += f'_seed{script_dict["seed"]}'

    for checkpoint in sys.argv[6:]:
        this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{checkpoint}'
        cycleGun.checkpoint = this_checkpoint
        label_dict = {}
        label_dict['fake'] = f'{label_prefix}_checkpoint{checkpoint}_fake'
        label_dict['cycled'] = f'{label_prefix}_checkpoint{checkpoint}_cycled'
        
        print(f"Rendering checkpoint {this_checkpoint}...")
        cycleGun.render_full(side_length=int(sys.argv[5]), side='A', cycle=(sys.argv[4].lower() == 'true'), label_dict=label_dict)#, crop_to_valid=True)
        print(f"Rendered!")
    
    print('Done.')