#%%
from glob import glob
import os
from silx import io
# import daisy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from natsort import natsorted
import napari

def int16_to_8(data):
    return (255 * (data / (2**16 - 1))).astype(np.uint8)

def int_to_float(data):
    return (data / (2**16 - 1)).astype(np.float16)

def get_scan_name(file):
    file = file.split(os.sep)[-1]
    return file.replace('.edf', '')

def load_data(files, reduce=False, as_float=True):
    data = []
    for f in trange(len(files)):
        file = io.open(files[f])
        raw = file['scan_0']['image']['data'].value
        if as_float:
            raw = int_to_float(raw)
        else:
            raw = int16_to_8(raw)
        data.append(raw)
    
    data = np.array(data)
    if reduce:
        data = getattr(data, reduce)(0)

    return data

def compile_scan(folder, save=True, raw=False):
    
    files = glob(os.path.join(folder, '*.edf'))
    files = natsorted(files)
    dark_files = [file for file in files if 'dark' in file]
    ref_files = [file for file in files if 'ref' in file]
    _ = [files.remove(dark_file) for dark_file in dark_files]
    _ = [files.remove(ref) for ref in ref_files]

    ref_files = [ref for ref in ref_files if 'HST' in ref]

    data = load_data(files)
    dark = load_data(dark_files, 'mean')
    ref = load_data(ref_files, 'mean')

    if save:
        np.save(os.path.join(folder, 'raw.npy'), data)
        np.save(os.path.join(folder, 'dark.npy'), dark)
        np.save(os.path.join(folder, 'ref.npy'), ref)

    if not raw:
        data /= ref
        data -= dark

    if save:        
        np.save(os.path.join(folder, 'data.npy'), data)
    
    return data, dark, ref

#%%
i = 100
folder = '/home/rhoadesj/Data/XNH/CBxs_lobV_overview_90nm_1_/'
raw, dark, ref = compile_scan(folder, save=False, raw=True)
raw = raw[i]

# %%
data, dark, ref = compile_scan(folder, save=False)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(raw, cmap='gray')
axs[0].set_title('Example Corrected Scan')
axs[0].axis('off')

axs[1].imshow(dark, cmap='gray')
axs[1].set_title('Dark Image')
axs[1].axis('off')

axs[2].imshow(ref, cmap='gray')
axs[2].set_title('Empty Beam')
axs[2].axis('off')

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[3].imshow(data[i], cmap='gray')
axs[3].set_title('Example Corrected Scan')
axs[3].axis('off')


#%%
ref_files = glob(os.path.join(folder, 'ref*.edf'))
refs = load_data(natsorted(ref_files))
len(refs)

#%%
nrows = 11
ncols = 4

fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
for r in range(nrows):
    for c in range(ncols):
        axs[r, c].imshow(refs[r*c + c], cmap='gray')
        axs[r, c].axis('off')
        axs[r, c].set_title(ref_files[r*c+c].split('/')[-1])

#%%
freqs = np.fft.fftfreq(ref.shape[-1])
ref_fft = np.fft.fft(ref)
data_fft = np.fft.fft(data[100])



# %%
viewer = napari.Viewer()
viewer.add_image(data, name='Corrected Scans')
