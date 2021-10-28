import daisy

#Specify source and destination
source_file = '/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr'
source_ds = 'volumes/raw'

destination_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/cb2_gcl1.n5'
destination_ds = 'volumes/raw'

offset = [2800, 520000, 744000] #IN WORLD UNITS (z,y,x)
shape = [44080, 96480, 96480] #IN WORLD UNITS (z,y,x)

#default options
chunk_size = 64
num_channels = 1
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }
num_workers = 30

#Load data
source = daisy.open_ds(source_file, source_ds)

#Prepare new dataset
total_roi = daisy.Roi(offset, shape)
write_size = source.voxel_size * chunk_size
destination = daisy.prepare_ds(
        destination_file,
        destination_ds,
        total_roi,
        source.voxel_size,
        source.dtype,
        write_size=write_size,
        num_channels=num_channels,
        compressor=compressor)

#Prepare saving function/variables
def save_chunk(read_roi, write_roi):
    destination.__setitem__(write_roi, source.__getitem__(read_roi))

chunk_roi = daisy.Roi([0,]*len(offset), write_size)

#Write data to new dataset
success = daisy.run_blockwise(
            total_roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit='shrink',
            num_workers=num_workers,
            max_retries=3)

if success:
    print(f'{total_roi} from {source_file}/{source_ds} written to {destination_file}/{destination_ds}')
else:
    print('Failed to save cutout.')
