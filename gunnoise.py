# !conda activate n2v

import zarr
import os
import matplotlib.pyplot as plt

import gunpowder as gp
import logging
# logging.basicConfig(level=logging.INFO)

# from this repo
# from segway.tasks.make_zarr_from_tiff import task_make_zarr_from_tiff_volume as tif2zarr
from boilerPlate import GaussBlur, Noiser

def bring_the_noise(src, pipeline, noise_order, noise_dict, noise_version=''):
    this_array = src
    noise_name = ''
    arrays = [src]
    for noise in noise_order:
        if noise_dict[noise]:
            if str(noise_dict[noise]).isnumeric():
                noise_name += noise + str(noise_dict[noise])
            elif isinstance(noise_dict[noise], bool):
                noise_name += noise

            new_array = gp.ArrayKey(noise_name.upper())
            
            if noise =='downX' and noise_dict[noise]:
                pipeline += gp.DownSample(this_array, (1, noise_dict[noise], noise_dict[noise]), new_array) # assumes zyx coordinates (and non-isometric)
            elif noise =='gaussBlur' and noise_dict[noise]:
                pipeline += GaussBlur(this_array, noise_dict[noise], new_array=new_array)
            elif noise =='gaussNoise' and noise_dict[noise]:
                pipeline += Noiser(this_array, new_array=new_array, mode='gaussian', var=noise_dict[noise])
            elif noise =='poissNoise' and noise_dict[noise]:
                pipeline += Noiser(this_array, new_array=new_array, mode='poisson')
            # elif noise =='deform' and noise_dict[noise]: # TODO: IMPLEMENT
            #     pipeline += ...
            
            noise_name += '_'
            this_array = new_array
            arrays.append(new_array)
    
    if noise_version == '':
        noise_name = noise_name[:-1]
    else:
        noise_name += noise_version
    return pipeline, arrays, noise_name

def noise_batch(samples,
    src_path,
    raw_name,
    noise_dict,
    noise_order,
    src_voxel_size=(40, 4, 4),
    check_every=250,
    stack_size=8,
    scan_size=(40, 512, 512)
    ):
    # assemble and run pipeline for each dataset
    for sample in samples:
        
        if noise_dict['downX']:
            dest_voxel_size = [src_voxel_size[s] * noise_dict['downX'] if s > 0 else src_voxel_size[s] for s in range(len(src_voxel_size))]
        else:
            dest_voxel_size = src_voxel_size
        src_voxel_size = gp.Coordinate(src_voxel_size)
        dest_voxel_size = gp.Coordinate(dest_voxel_size)

        # declare arrays to use in the pipeline
        raw = gp.ArrayKey('RAW') # raw data
        raw_spec = gp.ArraySpec(voxel_size=src_voxel_size, interpolatable=True)

        src = f'{src_path}{sample}/{sample}.zarr/volumes'
        zarr.open(src)
        source = gp.ZarrSource(    # add the data source
                src,  # the zarr container
                {raw: raw_name},  # which dataset to associate to the array key
                {raw: raw_spec}  # meta-information
        )

        # add normalization
        normalize = gp.Normalize(raw)

        proto_pipe = source + normalize

        pipeline, arrays, noise_name = bring_the_noise(raw, proto_pipe, noise_order, noise_dict)

        noisy = arrays[-1] # data noise added
        
        stack = gp.Stack(stack_size)

        # request matching the model input and output sizes
        scan_request = gp.BatchRequest()
        for array in arrays:
                scan_request.add(array, scan_size)

        scan = gp.Scan(scan_request, num_workers=os.cpu_count())

        # request an empty batch from scan
        request = gp.BatchRequest()

        # setup Cache
        cache = gp.PreCache()

        # get performance stats
        performance = gp.PrintProfilingStats(every=check_every)

        destination = gp.ZarrWrite(
                dataset_names = {noisy: noise_name},
                output_dir = f'{src_path}{sample}',
                output_filename = f'{sample}.zarr/volumes',
                #dataset_dtypes = {noisy: np.uint8} # save as 0-255 values (should match raw)
        )

        pipeline += (cache +
                stack + 
                destination + 
                scan + 
                performance)
        
        with gp.build(pipeline):
                print(f'Starting {noise_name} of {src}...')
                pipeline.request_batch(request)
                print(f'Brought the noise to {src}')

def test_noise(sample,
    src_path,
    raw_name,
    noise_dict,
    noise_order,
    src_voxel_size=(40, 4, 4),
    test_size=(40, 2048, 2048)
    ):

    # declare arrays to use in the pipeline
    raw = gp.ArrayKey('RAW') # raw data
    raw_spec = gp.ArraySpec(voxel_size=src_voxel_size, interpolatable=True)

    src = f'{src_path}{sample}/{sample}.zarr/volumes'
    zarr.open(src)
    source = gp.ZarrSource(    # add the data source
            src,  # the zarr container
            {raw: raw_name},  # which dataset to associate to the array key
            {raw: raw_spec}  # meta-information
    )

    # add normalization
    normalize = gp.Normalize(raw)

    # add a RandomLocation node to the pipeline to randomly select a sample
    random_location = gp.RandomLocation()

    proto_pipe = source + normalize + random_location

    pipeline, arrays, noise_name = bring_the_noise(raw, proto_pipe, noise_order, noise_dict)

    noisy = arrays[-1] # data noise added

    # request matching the model input and output sizes
    request = gp.BatchRequest()
    for array in arrays:
            request.add(array, test_size)

    with gp.build(pipeline):
            batch = pipeline.request_batch(request)
    
    fig, axs = plt.subplots(1,len(arrays), figsize=(40, 40*len(arrays)))
    for i, array in enumerate(arrays):
        axs[i].imshow(batch[array].data.squeeze(), cmap='gray')
        axs[i].set_title(array.identifier)