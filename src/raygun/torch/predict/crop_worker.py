#%%
import os
import daisy
import torch
import numpy as np
import logging
from time import sleep
logging.basicConfig(level=logging.INFO)

from raygun import load_system

def crop_worker(config_path, source_path, source_dataset, net_name, checkpoint=None):
    worker_id = client.worker_id
    logger = logging.getLogger(f'crop_worker_{worker_id}')
    client = daisy.Client()
    logger.info(f"Launching {worker_id}...")

    # Setup rendering pipeline
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
                                
                if torch.cuda.is_available():
                    out = out.cpu().numpy().astype(destination.dtype)
                else:
                    out = out.numpy().astype(destination.dtype)

                destination[this_write] = out
                del out

if __name__ == '__main__':
    dest_path = os.path.join(os.path.dirname(config_path), os.path.basename(source_path))
    dest_dataset = f"{source_dataset}_{net_name}_{checkpoint}"

    compressor = {  'id': 'blosc', 
                    'clevel': 3,
                    'cname': 'blosclz',
                    'blocksize': chunk_size
                    }

    destination = daisy.prepare_ds(
            self.src_path, 
            label_dict['fake'],
            total_roi,
            self.source.voxel_size,
            self.source.dtype,
            write_size=write_roi.get_shape() if not self.smooth else self.write_pad*2,
            compressor=compressor,
            delete=True,
            # force_exact_write_size=True
            )
        if needs_erase:
            self.destination[self.destination.roi] = 0
        
    
    # task = daisy.Task(
    #     'DummyTask',
    #     daisy.Roi((0,), (1000,)),
    #     daisy.Roi((0,), (10,)),
    #     daisy.Roi((1,), (8,)),
    #     process_function=lambda: crop_worker("[fancy options]"),
    #     num_workers=23
    # )
    # daisy.run_blockwise([task])
# %%
