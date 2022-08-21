# %%
from genericpath import exists
from glob import glob
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
# import pyphase
from vendor.EdfFile import EdfFile

def siemens_star(n, d, b, pad=None):
    """"Return Siemen's Star of arbitrary (delta and beta) refractive index set by d, b"""
    data = plt.imread('./Siemens_star.svg.png')
    data = jnp.expand_dims(jnp.expand_dims(jnp.float64(jnp.max(data, axis=-1) > 0), 0), -1) # 'h w -> b h w 1'
    if pad is not None:
        if pad is True:
            pad = data.shape[1] // 2
        data = np.pad(data, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    
    # delta and beta should be 4d Tensors
    delta = data * d # 1 minus the refractive index
    beta = data * b # the extinction coefficient

    # make 0's in delta correspond to empty space
    delta[delta == 0] = (1 - n)

    return delta, beta
# %%

def get_ESRF_angle(path, prefix, angle=0, distances=[1,2,3,4], recon_name='rectwopassdb9', include_refs=True, include_dark=True):
    if not path.endswith('/'): path += '/'
    if not prefix.endswith('_'): prefix += '_'
    suffix = str(angle).zfill(4) + '.edf'
    
    ims = []
    for distance in distances:
        this_prefix = prefix + str(distance) + '_'
        imEDF = EdfFile(path + this_prefix + '/' + this_prefix + suffix)
        ims.append(imEDF)
    out_dict = {'images': get_arr(ims), 'edfs': ims}

    if exists(path + prefix + '/' + prefix + recon_name + '_' + suffix):
        out_dict.update({'recon_image': get_arr([imEDF]), 'recon_edf': imEDF})

    if include_refs:
        out_dict.update(get_ESRF_refs(path, prefix, distances, include_dark))

    return out_dict

def get_ESRF_refs(path, prefix, distances=[1,2,3,4], include_dark=True):
    # TODO: Currently concatenates all ref images from before/after scans for all distances
    if not path.endswith('/'): path += '/'
    if not prefix.endswith('_'): prefix += '_'
    
    ims = []
    darks = []
    for distance in distances:
        this_prefix = prefix + str(distance) + '_'
        this_path = path + this_prefix + '/'
        ref_file_list = glob(this_path + 'refHST*.edf')
        ref_file_list.sort()
        for file in ref_file_list:
            imEDF = EdfFile(file)
            ims.append(imEDF)
        
        if include_dark and exists(this_path + 'dark.edf'):
            darks.append(EdfFile(this_path + 'dark.edf'))
    
    out_dict = {'empty_images': get_arr(ims), 'empty_edfs': ims}
    if include_dark:
        out_dict.update({'dark_images': get_arr(darks), 'dark_edfs': darks})

    return out_dict

# %%
def show_edf(filename):
    file = EdfFile(filename)
    plt.imshow(file.GetData(0))
    plt.colorbar()
    return file.GetHeader(0)

def get_arr(edfs):
    shape = edfs[0].GetData(0).shape
    images_arr = np.empty((len(edfs), shape[0], shape[1], 1))
    for i in range(len(edfs)):
        images_arr[i,...,0] = edfs[i].GetData(0)
    
    return images_arr
# %%
