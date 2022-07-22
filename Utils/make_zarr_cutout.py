import sys
import daisy
import numpy as np


def make_zarr_cutout(source_file,
                    source_ds,
                    destination_file,
                    destination_ds,
                    offset,
                    shape,
                    chunk_size = 64,
                    num_channels = 1,
                    compressor = {  'id': 'blosc', 
                                    'clevel': 3,
                                    'cname': 'blosclz',
                                    'blocksize': 64
                                    },
                    num_workers = 30
                ):

    #Load data
    source = daisy.open_ds(source_file, source_ds)
    if source.n_channel_dims != 0:
        num_channels = source.shape[0]

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
    
    return success


if __name__ == "__main__":
    #Specify source and destination
    source_file = sys.argv[1]
    source_ds = sys.argv[2]

    destination_file = sys.argv[3]
    destination_ds = sys.argv[4]

    #IN WORLD UNITS (z,y,x)
    offset = daisy.Coordinate(sys.argv[5].split(','))
    shape = daisy.Coordinate(sys.argv[6].split(','))

    make_zarr_cutout(source_file, source_ds, destination_file, destination_ds, offset, shape)

