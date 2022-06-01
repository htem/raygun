import argparse
import runpy
import numpy as np
import daisy
import torch
from numpy import uint8
from scipy.signal.windows import tukey
# import sys
# sys.path.append('/n/groups/htem/users/jlr54/raygun/')
# from CycleGAN import *


def render_tiled(script_path, 
                side, 
                src_path, 
                src_name, 
                checkpoints, 
                side_length=None, # needs to be multiple of 4
                cycle=False, 
                crop=0,
                smooth=True,
                num_workers=30, 
                alpha=0.5
                ):
    
    #Load Model
    script_dict = runpy.run_path(script_path)
    cycleGun = script_dict['cycleGun']
    if side_length is None:
        side_length = cycleGun.side_length

    #Set Dataset to Render
    source = daisy.open_ds(src_path, src_name)
    side = side.upper()
    if type(checkpoints) is not list:
        checkpoints = [checkpoints]    

    if len(cycleGun.model_name.split('-')) > 1:
        parts = cycleGun.model_name.split('-')
        label_prefix = parts[0] + parts[1][1:]
    else:
        label_prefix = cycleGun.model_name
    label_prefix += f'_seed{script_dict["seed"]}'

    read_size = side_length
    read_roi = daisy.Roi([0,0,0], source.voxel_size * read_size)
    write_size = read_size

    if crop:
        write_size -= crop*2

    if smooth:
        window = torch.cuda.FloatTensor(tukey(write_size, alpha=alpha))
        window = (window[:, None, None] * window[None, :, None]) * window[None, None, :]
        chunk_size = int(write_size * (alpha / 2))
        write_pad = (source.voxel_size * write_size * (alpha / 2)) / 2
        write_size -= write_size * (alpha / 2)
        write_roi = daisy.Roi(source.voxel_size * crop + write_pad, source.voxel_size * write_size)
    else:
        write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)
        chunk_size = int(write_size)

    num_channels = 1
    compressor = {  'id': 'blosc', 
                    'clevel': 3,
                    'cname': 'blosclz',
                    'blocksize': chunk_size
                    }
    
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
        label_dict['fake'] = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}'
        label_dict['cycled'] = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}{other_net}'
        
        print(f"Preparing dataset {label_dict['fake']} at {src_path}")
        destination = daisy.prepare_ds(
            src_path, 
            label_dict['fake'],
            source.roi,
            source.voxel_size,
            source.dtype,
            write_size=write_roi.get_shape() if not smooth else write_pad*2,
            num_channels=num_channels,
            compressor=compressor,
            delete=True,
            # force_exact_write_size=True
            )

        #Prepare saving function/variables
        def save_chunk(block:daisy.Roi):
            try:
                write_roi = block.write_roi
                data = source.to_ndarray(block.read_roi)
                data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
                data -= np.iinfo(source.dtype).min
                data /= np.iinfo(source.dtype).max
                data *= 2
                data -= 1
                out = generator(data).detach().squeeze()
                if crop:
                    out = out[crop:-crop, crop:-crop, crop:-crop]
                out += 1.0
                out /= 2
                if smooth:
                    out *= window
                    write_roi = write_roi.grow(write_pad, write_pad)
                out *= 255 #TODO: This is written assuming dtype = uint8
                out = out.cpu().numpy().astype(uint8)
                destination[write_roi] = destination[write_roi].to_ndarray() + out
                return 0 # success
            except:
                return 1 # error

        success = daisy.run_blockwise(
            source.roi,
            read_roi,
            write_roi,
            process_function=save_chunk,
            read_write_conflict=True,
            # fit='shrink',
            num_workers=num_workers,
            max_retries=2)

        if success:
            print(f'{source.roi} from {src_path}/{src_name} rendered and written to {src_path}/{label_dict["fake"]}')
        else:
            print('Failed to save cutout.')
        
        if cycle:
            ...#TODO: read from the one just rendered and write to a new zarr
    
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
    ap.add_argument("--crop", type=int, help='How much to crop network outputs by (Default: 0).', default=0)
    ap.add_argument("--smooth", type=bool, help='Whether to blend edges of network outputs (e.g. "false" or "true").', default=True)
    ap.add_argument("--num_workers", type=int, help='How many workers to run in parallel.', default=30)
    config = ap.parse_args()
    
    render_tiled(**vars(config))
    #TODO: ADD MINIMAL VERSION