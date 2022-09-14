#%%
from importlib import import_module
import os
from subprocess import Popen
import sys
import tempfile
import daisy
import logging
logging.basicConfig(level=logging.INFO)

from raygun import load_system, read_config

#%%
def predict(render_config_path=None):    
    if render_config_path is None:
        render_config_path = sys.argv[1]

    logger = logging.getLogger(__name__)
    logger.info('Loading prediction config...')

    render_config = read_config(render_config_path)

    config_path = render_config['config_path']
    train_config = read_config(config_path)
    source_path = render_config['source_path']
    source_dataset = render_config['source_dataset']
    net_name = render_config['net_name']
    checkpoint = render_config['checkpoint']
    # compressor = render_config['compressor']
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

    destination = daisy.prepare_ds(
            dest_path, 
            dest_dataset,
            source.data_roi,
            source.voxel_size,
            source.dtype,
            write_size=write_roi.get_shape(),
            # compressor=compressor,
            )                

    # with tempfile.TemporaryDirectory() as temp_dir:
    #     cur_dir = os.getcwd()
    #     os.chdir(temp_dir)
    #     print(f'Executing in {os.getcwd()}')

    if 'launch_command' in render_config.keys():
        process_function = lambda: Popen([*render_config['launch_command'].split(' '), 
                                    "python", 
                                    import_module('.'.join(['raygun', train_config['framework'], 'predict', 'worker'])).__file__,
                                    render_config_path])

    else:
        worker = getattr(import_module('.'.join(['raygun', train_config['framework'], 'predict', 'worker'])), 'worker')
        process_function = lambda: worker(render_config_path)

    task = daisy.Task(
            os.path.basename(render_config_path).rstrip('.json'),
            total_roi=source.data_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            read_write_conflict=True,
            fit='shrink',
            num_workers=num_workers,
            max_retries=max_retries,
            process_function=process_function
        )

    logger.info('Running blockwise prediction...')
    daisy.run_blockwise([task])

    logger.info('Saving viewer script...')
    view_script = os.path.join(os.path.dirname(config_path), f"view_{os.path.basename(source_path).rstrip('.n5').rstrip('.zarr')}.ng")
    
    if not os.path.exists(view_script):
        with open(view_script, "w") as f:
            f.write(f"neuroglancer -f {source_path} -d {source_dataset} -f {dest_path} -d {dest_dataset} ")

    else:
        with open(view_script, "a") as f:
            f.write("{dest_dataset} ")

    logger.info('Done.')

        # os.chdir(cur_dir)
# %%
if __name__ == '__main__':
    predict(sys.argv[1])
