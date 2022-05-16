import sys
import runpy
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from CycleGAN import *

def render(path, raw_name, target_roi, script_path, checkpoints, net='netG1', as_half=True):
    #Get Data to Render
    source = daisy.open_ds(path, raw_name)
    data = source.to_ndarray(target_roi)

    #Put on cuda and scale to [-1,1]
    if as_half:
        data = torch.cuda.HalfTensor(data).unsqueeze(0).unsqueeze(0)
    else:
        data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
    data -= data.min()
    data /= data.max()
    data *= 2
    data -= 1

    #Load Model
    script_dict = runpy.run_path(script_path)
    cycleGun = script_dict['cycleGun']

    if len(cycleGun.model_name.split('-')) > 1:
        parts = cycleGun.model_name.split('-')
        label_prefix = parts[0] + parts[1][1:]
    else:
        label_prefix = cycleGun.model_name
    label_prefix += f'_seed{script_dict["seed"]}'

    successes = 0
    for checkpoint in checkpoints:
        out_name = f'volumes/{label_prefix}_checkpoint{checkpoint}_{net}'
        this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{checkpoint}'
        cycleGun.checkpoint = this_checkpoint
        cycleGun.load_saved_model(this_checkpoint)
        generator = getattr(cycleGun.model, net).cuda()
        generator.eval()
        if as_half:
            generator.to(torch.half)
        
        print(f"Rendering checkpoint {this_checkpoint}...")
        out = generator(data).detach().squeeze()
        out += 1.0
        out /= 2
        out *= 255
        out = out.cpu().numpy().astype(source.dtype)
        print(f"Rendered!")
        
        #Write result
        out_ds = daisy.Array(out, target_roi, source.voxel_size)

        chunk_size = 64
        num_channels = 1
        compressor = {  'id': 'blosc', 
                        'clevel': 3,
                        'cname': 'blosclz',
                        'blocksize': chunk_size
                        }
        num_workers = 30
        write_size = out_ds.voxel_size * chunk_size
        chunk_roi = daisy.Roi([0,]*len(target_roi.get_offset()), write_size)

        destination = daisy.prepare_ds(
            path, 
            out_name,
            target_roi,
            out_ds.voxel_size,
            out.dtype,
            write_size=write_size,
            write_roi=chunk_roi,
            num_channels=num_channels,
            compressor=compressor)

        #Prepare saving function/variables
        def save_chunk(block:daisy.Roi):
            try:
                destination.__setitem__(block.write_roi, out_ds.__getitem__(block.read_roi))
                return 0 # success
            except:
                return 1 # error
                
        #Write data to new dataset
        success = daisy.run_blockwise(
                    target_roi,
                    chunk_roi,
                    chunk_roi,
                    process_function=save_chunk,
                    read_write_conflict=False,
                    fit='shrink',
                    num_workers=num_workers,
                    max_retries=2)

        if success:
            print(f'{target_roi} from {path}/{raw_name} rendered and written to {path}/{out_name}')
            successes += 1
        else:
            print('Failed to save cutout.')
        
    return successes 

if __name__ == '__main__':
    # python CBvRenderNetOutputs.py <Path to Training Script> <Source Path> <Source Name> <net> <pad> <checkpoint1> ... <checkpointN>
    
    script_path = sys.argv[1]
    path = sys.argv[2]
    raw_name = sys.argv[3]
    net = sys.argv[4]
    pad = int(sys.argv[5])
    checkpoints = sys.argv[6:]
    if type(checkpoints) is not list:
        checkpoints = [checkpoints]

    source = daisy.open_ds(path, raw_name)
    target_roi = source.data_roi.grow(source.voxel_size * -pad, source.voxel_size * -pad)    

    successes = render(path, raw_name, target_roi, script_path, checkpoints, net=net)
    
    print(f'Done. {successes} successes out of {len(checkpoints)} attempted.')