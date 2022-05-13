# %%
#IMPORTS
import webknossos as wk
import wkw
from time import gmtime, strftime
import zipfile
import daisy
import tempfile
from glob import glob
import os

#%%
#ARGS
annotation_ID = '626fd7b6010000a80079d0d9'
save_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/annotations/'

zarr_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
raw_name ='volumes/raw_30nm'

#Defaults:
wk_url = 'http://catmaid2.hms.harvard.edu:9000'
annotation_url_prefix = 'http://catmaid2.hms.harvard.edu:9000/annotations/Explorational/'
wk_token = "Q9OpWh1PPwHYfH9BsnoM2Q"
gt_name_prefix = 'volumes/'

# %%
# Load Zarr dataset and annotation
ds = daisy.open_ds(zarr_path, raw_name)
with wk.webknossos_context(token="Q9OpWh1PPwHYfH9BsnoM2Q", url='http://catmaid2.hms.harvard.edu:9000'):

    annotation = wk.Annotation.download(
        "http://catmaid2.hms.harvard.edu:9000/annotations/Explorational/" + annotation_ID
    )
# %%
time_str = strftime("%Y%m%d", gmtime())
annotation_name = f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'

#%%
zip_path = save_path + annotation_name + '.zip'
annotation.save(zip_path)

# %%
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(save_path + annotation_name)

with zipfile.ZipFile(save_path + annotation_name + '/data_0_Volume.zip', 'r') as zip_ref:
    zip_ref.extractall(save_path + annotation_name + '/data_0_Volume')

# %%
segmentations = wkw.Dataset.open(save_path + annotation_name + '/data_0_Volume/1')
data = segmentations.read([312, 312, 312], [400, 400, 400])



# %%
#FUNCTION
def wkw_seg_to_zarr(
        annotation_ID = '626fd7b6010000a80079d0d9',
        save_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/annotations/',
        zarr_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5',
        raw_name ='volumes/raw_30nm',
        wk_url = 'http://catmaid2.hms.harvard.edu:9000',
        annotation_url_prefix = 'http://catmaid2.hms.harvard.edu:9000/annotations/Explorational/',
        wk_token = "Q9OpWh1PPwHYfH9BsnoM2Q",
        gt_name_prefix = 'volumes/',
    ):
    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'

    print(f"Downloading and saving {annotation_url_prefix + annotation_ID} as {annotation_ID}...")

    with wk.webknossos_context(token=wk_token, url=wk_url):
        annotation = wk.Annotation.download(
            annotation_url_prefix + annotation_ID
        )
    zip_path = save_path + annotation_name + '.zip'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annotation.save(zip_path)

    # Extract zip file
    zf = zipfile.ZipFile(zip_path)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        zipped_datafile = glob(tempdir + '/data*.zip')
        print(f"Opening {zipped_datafile}...")
        zf_data = zipfile.ZipFile(zipped_datafile)
        with tempfile.TemporaryDirectory() as zf_data_tmpdir:
            zf_data.extractall(zf_data_tmpdir)

            # Open the WKW dataset (as the `1` folder)
            print(f"Opening {zf_data_tmpdir + '/1'}...")
            dataset = wkw.Dataset.open(zf_data_tmpdir + '/1')
            data = dataset.read(ds.roi.get_offset() / ds.voxel_size, ds.roi.get_shape() / ds.voxel_size).squeeze()
            
            # Save annotations to zarr
            gt_name = f'{gt_name_prefix}gt_{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
            
            target_roi = ds.roi
            gt_array = daisy.Array(data, ds.roi, ds.voxel_size)

            
            chunk_size = ds.chunk_shape
            num_channels = 1
            compressor = {  'id': 'blosc', 
                            'clevel': 3,
                            'cname': 'blosclz',
                            'blocksize': chunk_size
                            }
            num_workers = 30
            write_size = gt_array.voxel_size * chunk_size
            chunk_roi = daisy.Roi([0,]*len(target_roi.get_offset()), write_size)

            destination = daisy.prepare_ds(
                zarr_path, 
                gt_name,
                target_roi,
                gt_array.voxel_size,
                data.dtype,
                write_size=write_size,
                write_roi=chunk_roi,
                num_channels=num_channels,
                compressor=compressor)

            #Prepare saving function/variables
            def save_chunk(block:daisy.Roi):
                try:
                    destination.__setitem__(block.write_roi, gt_array.__getitem__(block.read_roi))
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
                print(f'{target_roi} from {annotation_name} written to {zarr_path}/{gt_name}')
            else:
                print('Failed to save annotation layer.')