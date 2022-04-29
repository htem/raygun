#%% 
import daisy

source_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
source_ds = 'volumes/affs0_30nm'
source = daisy.open_ds(source_file, source_ds)

destination_file = source_file
destination_ds = 'volumes/affs0min_30nm'


chunk_size = 64
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }
num_workers = 30
total_roi = source.roi
write_size = source.voxel_size * chunk_size
chunk_roi = daisy.Roi([0,]*len(total_roi.get_offset()), write_size)
num_channels = 1

destination = daisy.prepare_ds(
        destination_file,
        destination_ds,
        total_roi,
        source.voxel_size,
        source.dtype,
        write_size=write_size,
        write_roi=chunk_roi,
        num_channels=num_channels,
        compressor=compressor)

def save_chunk(block:daisy.Roi):
    try:
        destination.__setitem__(block.write_roi, source.to_ndarray(block.read_roi).min(axis=0))
        return 0 # success
    except:
        return 1 # error
        
#%%
#Write data to new dataset
success = daisy.run_blockwise(
            total_roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit='shrink',
            num_workers=num_workers,
            max_retries=2)
# %%
if success:
    print(f'{total_roi} from {source_file}/{source_ds} written to {destination_file}/{destination_ds}')
else:
    print('Failed to save cutout.')