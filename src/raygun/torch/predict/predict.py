#%%
import os
import sys
import tempfile
import daisy
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from raygun import load_system, read_config

def worker(render_config_path):
    client = daisy.Client()
    worker_id = client.worker_id
    logger = logging.getLogger(f'crop_worker_{worker_id}')
    logger.info(f"Launching {worker_id}...")

    # Setup rendering pipeline
    render_config = read_config(render_config_path)

    config_path = render_config['config_path']
    source_path = render_config['source_path']
    source_dataset = render_config['source_dataset']
    net_name = render_config['net_name']
    checkpoint = render_config['checkpoint']
    crop = render_config['crop']
    
    system = load_system(config_path)
    
    if not os.path.exists(str(checkpoint)):
        checkpoint_path = os.path.join(os.path.dirname(config_path), system.checkpoint_basename.lstrip('./') + f'_checkpoint_{checkpoint}')
        
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
    
    else:
            checkpoint_path = None

    system.load_saved_model(checkpoint_path)
    net = getattr(system.model, net_name)
    net.eval()
    if torch.cuda.is_available():
        net.to("cuda") #TODO pick best option

    del system

    source = daisy.open_ds(source_path, source_dataset)

    dest_path = os.path.join(os.path.dirname(config_path), os.path.basename(source_path))
    dest_dataset = f"{source_dataset}_{net_name}_{checkpoint}"
    destination = daisy.open_ds(dest_path, dest_dataset, 'a')

    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            
            else:                 
                this_write = block.write_roi
                data = source.to_ndarray(block.read_roi)
                if torch.cuda.is_available():
                    data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
                else:
                    data = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)

                data -= np.iinfo(source.dtype).min
                data /= np.iinfo(source.dtype).max
                out = net(data).detach().squeeze()            
                del data

                if crop:
                    out = out[crop:-crop, crop:-crop, crop:-crop]
                
                out *= np.iinfo(destination.dtype).max
                out = torch.clamp(out, np.iinfo(destination.dtype).min, np.iinfo(destination.dtype).max)
                                
                if torch.cuda.is_available():
                    out = out.cpu().numpy().astype(destination.dtype)
                else:
                    out = out.numpy().astype(destination.dtype)

                destination[this_write] = out
                del out

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.info('Loading prediction config...')

    render_config_path = sys.argv[1]
    render_config = read_config(render_config_path)

    config_path = render_config['config_path']
    source_path = render_config['source_path']
    source_dataset = render_config['source_dataset']
    net_name = render_config['net_name']
    checkpoint = render_config['checkpoint']
    compressor = render_config['compressor']
    read_size = render_config['read_size']
    crop = render_config['crop']
    num_workers = render_config['num_workers']
    max_retries = render_config['max_retries']
    
    dest_path = os.path.join(os.path.dirname(config_path), os.path.basename(source_path))
    dest_dataset = f"{source_dataset}_{net_name}_{checkpoint}"

    source = daisy.open_ds(source_path, source_dataset)

    read_roi = daisy.Roi([0,0,0], source.voxel_size * read_size)
    write_size = read_size - crop*2
    write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)
    chunk_size = int(write_size)

    destination = daisy.prepare_ds(
            dest_path, 
            dest_dataset,
            source.data_roi,
            source.voxel_size,
            source.dtype,
            write_size=write_roi.get_shape(),
            compressor=compressor,
            )        
    
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     cur_dir = os.getcwd()
    #     os.chdir(temp_dir)
    #     print(f'Executing in {os.getcwd()}')

    task = daisy.Task(
            os.path.basename(render_config_path).rstrip('.json'),
            total_roi=source.data_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            read_write_conflict=True,
            fit='shrink',
            num_workers=num_workers,
            max_retries=max_retries,
            process_function=lambda: worker(render_config_path)
        )

    logger.info('Running blockwise prediction...')
    daisy.run_blockwise([task])
    logger.info('Done.')

        # os.chdir(cur_dir)