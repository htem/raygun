import daisy

#Specify source and destination
source_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
source_ds = 'volumes/raw_mipmap/s3_rechunked'

destination_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/cb2/myelWM1.n5'
destination_ds = 'volumes/raw_mipmap3'


offset = [8720, 510016, 580000] #IN WORLD UNITS (z,y,x)
shape = [20000, 20000, 20000] #IN WORLD UNITS (z,y,x)

#default options (shouldn't need to change)
chunk_size = 64
num_channels = 1
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }
num_workers = 30


#===============================================BELOW RUNS TASK


#Load data
source = daisy.open_ds(source_file, source_ds)

#Prepare new dataset
total_roi = daisy.Roi(offset, shape)
write_size = source.voxel_size * chunk_size
chunk_roi = daisy.Roi([0,]*len(offset), write_size)
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

#Prepare saving function/variables
def save_chunk(block:daisy.Roi):
    try:
        destination.__setitem__(block.write_roi, source.__getitem__(block.read_roi))
        return 0 # success
    except:
        return 1 # error
        

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

if success:
    print(f'{total_roi} from {source_file}/{source_ds} written to {destination_file}/{destination_ds}')
else:
    print('Failed to save cutout.')
