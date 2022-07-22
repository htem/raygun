import argparse
import os
import runpy
import numpy as np
import daisy
import torch
from scipy.signal.windows import tukey
# import sys
# sys.path.append('/n/groups/htem/users/jlr54/raygun/')
# from CycleGAN import *

import logging
logging.basicConfig(level=logging.INFO)

#Declare globals for daisy processes:
global window
global generator
global save_chunk
global task


def copy_pre_rendered(new_ds, old_ds, rw_roi, num_workers=30):
    def save_chunk(block:daisy.Roi):
        # try:
        data = old_ds.to_ndarray(block.read_roi)
        #make compatible across daisy versions if necessary
        if old_ds.n_channel_dims < new_ds.n_channel_dims:
            for i in range(new_ds.n_channel_dims - old_ds.n_channel_dims):
                data = np.expand_dims(data, axis=0)                
        elif old_ds.n_channel_dims > new_ds.n_channel_dims:
            for i in range(old_ds.n_channel_dims - new_ds.n_channel_dims):
                a = 0
                while data.shape[a] > 1:
                    a += 1
                    if a > data.ndim:
                        raise ValueError('Too few singleton channel dimensions to be compatible with new dataset.')
                data = np.squeeze(data, axis=a)        

        new_ds.__setitem__(block.write_roi, data)
        #     return 0 # success
        # except:
        #     return 1 # error
    
    task = daisy.Task(
            f'{__name__}---copying_prerendered',
            old_ds.roi,
            rw_roi,
            rw_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            # fit='shrink',
            num_workers=num_workers,
            max_retries=2)
    success = daisy.run_blockwise([task])
    if success:
        print(f'Copied pre-rendered data.')
    else:
        print('Failed to save cutout.')
    return success

def render_tiled(script_path, 
                side, 
                src_path, 
                src_name, 
                checkpoints, 
                side_length=None, # needs to be multiple of 4
                cycle=False, 
                crop=0,
                total_roi_crop=0,
                smooth=True,
                num_workers=30, 
                alpha=0.5,
                pre_rendered_suffix=None,
                ds_suffix='',
                require_cuda=False
                ):
    logger = logging.getLogger(__name__)
    src_path_name = src_path.split('/')[-1][:-3]
    
    cuda_available = False # torch.cuda.is_available() #TODO: Get CUDA working
    if not cuda_available:
        logger.warning('Cuda not available.')
        if require_cuda:
            print('Failed: cuda required and not available.')
            return
    else:
        torch.multiprocessing.set_start_method('spawn', force=True)

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

    read_size = side_length #e.g. 64
    read_roi = daisy.Roi([0,0,0], source.voxel_size * read_size)
    write_size = read_size

    if crop: #e.g. 16
        write_size -= crop*2 #e.g. --> 32

    if smooth:
        if cuda_available:
            window = torch.cuda.FloatTensor(tukey(write_size, alpha=alpha, sym=False))
        else:
            window = torch.FloatTensor(tukey(write_size, alpha=alpha, sym=False)).share_memory_()
        window = (window[:, None, None] * window[None, :, None]) * window[None, None, :]
        chunk_size = int(write_size * (alpha / 2)) #e.g. 8
        write_pad = (source.voxel_size * write_size * (alpha / 2)) / 2  #e.g. voxel_size * 4
        write_size -= write_size * (alpha / 2) #e.g. --> 24
        write_roi = daisy.Roi(source.voxel_size * crop + write_pad, source.voxel_size * write_size)
    else:
        write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)
        chunk_size = int(write_size)

    compressor = {  'id': 'blosc', 
                    'clevel': 3,
                    'cname': 'blosclz',
                    'blocksize': chunk_size
                    }
    
    net = 'netG1' if side == 'A' else 'netG2'
    other_net = 'netG1' if side == 'B' else 'netG2'

    for checkpoint in checkpoints:        
                      
        this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{checkpoint}'
        logger.info(f"Loading checkpoint {this_checkpoint}...")
        cycleGun.checkpoint = this_checkpoint
        cycleGun.load_saved_model(this_checkpoint, cuda_available)
        if cuda_available:
            generator = getattr(cycleGun.model, net).cuda()
        else:
            generator = getattr(cycleGun.model, net).cpu()
        generator.eval()
        logger.info(f"Rendering checkpoint {this_checkpoint}...")

        label_dict = {}
        label_dict['fake'] = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}{ds_suffix}'
        label_dict['cycled'] = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}{other_net}{ds_suffix}'
        
        logger.info(f"Preparing dataset {label_dict['fake']} at {src_path}")
        
        total_roi = source.roi
        if total_roi_crop:
            total_roi = total_roi.grow(source.voxel_size * -total_roi_crop, source.voxel_size * -total_roi_crop)

        needs_erase = os.path.exists(f'{src_path}/{label_dict["fake"]}')
        destination = daisy.prepare_ds(
            src_path, 
            label_dict['fake'],
            total_roi,
            source.voxel_size,
            source.dtype,
            write_size=write_roi.get_shape() if not smooth else write_pad*2,
            compressor=compressor,
            delete=True,
            # force_exact_write_size=True
            )
        if needs_erase:
            logger.info(f"Erasing existing dataset...")
            destination[destination.roi] = 0
        
        if pre_rendered_suffix is not None:
            pre_rendered_ds = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}{pre_rendered_suffix}'
            pre_rendered_src = daisy.open_ds(src_path, pre_rendered_ds)
            #copy pre-rendered into new ds
            success = copy_pre_rendered(destination, pre_rendered_src, read_roi, num_workers=num_workers)
            if not success:
                logger.info(f'Failed to copy pre-rendered data for {src_path}/{label_dict["fake"]}')
                continue
            pad = pre_rendered_src.voxel_size * -chunk_size # -write_size
            pre_rendered_roi = pre_rendered_src.roi.grow(pad, pad)
        else:
            pre_rendered_roi = None

        #Prepare saving function/variables
        def save_chunk(block:daisy.Roi):
            if pre_rendered_roi is not None and pre_rendered_roi.contains(block.write_roi):
                logger.info(f'Chunk for block ID {block.block_id} pre-rendered.')
                return 0
                
            logger.info(f'Attempting to save chunk for block ID {block.block_id}...')
            # try:
            this_write = block.write_roi
            logger.debug(f'Loading data for block ID {block.block_id}...')
            data = source.to_ndarray(block.read_roi)
            if cuda_available:
                logger.debug(f'Putting data on GPU for block ID {block.block_id}...')
                data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
            else:
                data = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)
            logger.debug(f'Normalizing data for block ID {block.block_id}...')
            data -= np.iinfo(source.dtype).min
            data /= np.iinfo(source.dtype).max
            data *= 2
            data -= 1
            logger.debug(f'Getting network output for block ID {block.block_id}...')
            out = generator(data).detach().squeeze()
            del data
            if crop:
                logger.debug(f'Cropping for block ID {block.block_id}...')
                out = out[crop:-crop, crop:-crop, crop:-crop]
            logger.debug(f'Normalizing output for block ID {block.block_id}...')
            out += 1.0
            out /= 2
            out *= 255 #TODO: This is written assuming dtype = np.uint8
            if smooth:
                logger.debug(f'Smoothing edges for block ID {block.block_id}...')
                out *= window
                logger.debug(f'Adjusting write ROI for block ID {block.block_id}...')
                this_write = this_write.grow(write_pad, write_pad)
            if cuda_available:
                logger.debug(f'Pulling to CPU for block ID {block.block_id}...')
                out = out.cpu().numpy()#.astype(np.uint8)
            else:
                out = out.numpy()#.astype(np.uint8)
            logger.debug(f'Getting existing data and summing for block ID {block.block_id}...')
            logger.info(f'Writing for block ID {block.block_id}...')
            destination[this_write] = np.uint8(destination.to_ndarray(this_write) + out)
            del out
            #     return 0 # success
            # except:
            #     return 1 # error

        task = daisy.Task(
            f'{__name__}---{src_path_name}-{label_dict["fake"]}',
            total_roi,
            read_roi,
            write_roi,
            process_function=save_chunk,
            read_write_conflict=True,
            # fit='shrink',
            num_workers=num_workers,
            max_retries=2)
        success = daisy.run_blockwise([task])

        if success:
            logger.info(f'{source.roi} from {src_path}/{src_name} rendered and written to {src_path}/{label_dict["fake"]}')
        else:
            logger.info('Failed to save cutout.')
        
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