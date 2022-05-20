import argparse
import sys
import runpy
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from CycleGAN import *

_def_render_tiled(source, )

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
    source = daisy.open_ds(src_path, src_name)
    side = side.upper()
    if type(checkpoints) is not list:
        checkpoints = [checkpoints]

    chunk_size = 64 # TODO: FIX
    num_channels = 1
    compressor = {  'id': 'blosc', 
                    'clevel': 3,
                    'cname': 'blosclz',
                    'blocksize': chunk_size
                    }
    write_size = source.voxel_size * chunk_size
    chunk_roi = daisy.Roi([0,0,0], write_size)

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
        print(f"Loading checkpoint {this_checkpoint}...")
        cycleGun.checkpoint = this_checkpoint
        cycleGun.load_saved_model(this_checkpoint)
        generator = getattr(cycleGun.model, net).cuda()
        generator.eval()
        print(f"Rendering checkpoint {this_checkpoint}...")

        label_dict = {}
        label_dict['fake'] = f'{label_prefix}_checkpoint{checkpoint}_{net}'
        label_dict['cycled'] = f'{label_prefix}_checkpoint{checkpoint}_{net}{other_net}'
        
        destination = daisy.prepare_ds(
            src_path, 
            label_dict['fake'],
            source.roi,
            source.voxel_size,
            source.dtype,
            write_size=write_size,
            write_roi=chunk_roi,
            num_channels=num_channels,
            compressor=compressor)

        #Prepare saving function/variables
        def save_chunk(block:daisy.Roi):
            try:
                data = source.to_ndarray(block.read_roi)
                data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
                data -= data.min()
                data /= data.max()
                data *= 2
                data -= 1
                out = generator(data).detach().squeeze()
                out += 1.0
                out /= 2
                out *= 255
                out = out.cpu().numpy().astype(source.dtype)
                #TODO: APPLY SMOOTH
                destination.__setitem__(block.write_roi, out)
                return 0 # success
            except:
                return 1 # error

        success = daisy.run_blockwise(
            source.roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit='shrink',
            num_workers=num_workers,
            max_retries=2)

        if success:
            print(f'{source.roi} from {src_path}/{src_name} rendered and written to {src_path}/{label_dict["fake"]}')
        else:
            print('Failed to save cutout.')
        
        if cycle:
            ...#TODO
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