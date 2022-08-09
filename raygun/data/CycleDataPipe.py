import math
import daisy
import gunpowder as gp
import numpy as np

from raygun.data import BaseDataPipe

class CycleDataPipe(BaseDataPipe):
    def __init__(self, id, src, ndims, common_voxel_size=None, interp_order=None):
        self.id = id        # id needed to ensure unique ArrayKeys
        self.src = src     
        self.ndims = ndims
        self.common_voxel_size = common_voxel_size
        self.interp_order = interp_order

        self.src_voxel_size = daisy.open_ds(self.src.path, self.src.real_name).voxel_size
        
        # declare arrays to use in the pipelines
        array_names = ['real', 
                        'fake', 
                        'cycled']
        if 'mask_name' in src.keys(): 
            array_names += ['mask']
            self.masked = True
        else:
            self.masked = False
                
        for array in array_names:
            if 'fake' in array:
                other_side = ['A','B']
                other_side.remove(id)
                array_name = array + '_' + other_side[0]
            else:
                array_name = array + '_' + id
            array_key = gp.ArrayKey(array_name.upper())
            setattr(self, array, array_key) # add ArrayKeys to object
            self.arrays += [array_key]

            #add normalizations and scaling, if appropriate        
            if 'mask' not in array:            
                setattr(self, 'scaletanh2img_'+array, gp.IntensityScaleShift(array_key, 0.5, 0.5))
            elif 'real' in array:                        
                setattr(self, 'normalize_'+array, gp.Normalize(array_key))
                setattr(self, 'scaleimg2tanh_'+array, gp.IntensityScaleShift(array_key, 2, -1))
        
        #Setup sources and resampling nodes
        if common_voxel_size is not None and common_voxel_size != self.src_voxel_size:
            self.real_src = gp.ArrayKey(f'REAL_{id}_SRC')
            self.resample = gp.Resample(self.real_src, common_voxel_size, self.real, ndim=self.ndims, interp_order=interp_order)
            if self.masked: 
                self.mask_src = gp.ArrayKey(f'MASK_{id}_SRC')
                self.resample += gp.Resample(self.mask_src, common_voxel_size, self.mask, ndim=self.ndims, interp_order=interp_order)
        else:            
            self.real_src = self.real
            self.resample = None
            if self.masked: 
                self.mask_src = self.mask

        # setup data sources
        if 'out_path' in src.keys():
            self.out_path = src['out_path']        
        self.src_names = {self.real_src: self.src.real_name}
        self.src_specs = {self.real_src: gp.ArraySpec(interpolatable=True, voxel_size=self.src_voxel_size)}
        if self.masked: 
            self.mask_name = src['mask_name']
            self.src_names[self.mask_src] = self.mask_name
            self.src_specs[self.mask_src] = gp.ArraySpec(interpolatable=False)

        if self.src.path.endswith('.zarr') or self.src.path.endswith('.n5'):
            self.source = gp.ZarrSource(    # add the data source
                        self.src.path,  
                        self.src_names,  # which dataset to associate to the array key
                        self.src_specs  # meta-information
            )
        elif self.src.path.endswith('.h5') or self.src.path.endswith('.hdf'):
            self.source = gp.Hdf5Source(    # add the data source
                        self.src.path,  
                        self.src_names,  # which dataset to associate to the array key
                        self.src_specs  # meta-information
            )
        else:
            raise NotImplemented(f'Datasource type of {self.src.path} not implemented yet. Feel free to contribute its implementation!')
        
        # setup rejections
        self.reject = None
        if self.masked:
            self.reject = gp.Reject(mask = self.mask_src, min_masked=0.999)

        if 'min_coefvar' in src.keys() and src['min_coefvar']:
            if self.reject is None:
                self.reject = gp.RejectConstant(self.real_src, min_coefvar = src['min_coefvar'])
            else:
                self.reject += gp.RejectConstant(self.real_src, min_coefvar = src['min_coefvar'])

        self.augment_axes = list(np.arange(3)[-self.ndims:])
        self.augment = gp.SimpleAugment(mirror_only = self.augment_axes, transpose_only = self.augment_axes)
        self.augment += self.normalize_real
        self.augment += self.scaleimg2tanh_real
        self.augment += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
                    control_point_spacing=100, # self.side_length//2,
                    # jitter_sigma=(5.0,)*self.ndims,
                    jitter_sigma=(0., 5.0, 5.0,)[-self.ndims:],
                    rotation_interval=(0, math.pi/2),
                    subsample=4,
                    spatial_dims=self.ndims
        )

        # add "channel" dimensions if neccessary, else use z dimension as channel
        if self.ndims == len(self.common_voxel_size):
            self.unsqueeze = gp.Unsqueeze([self.real])
        else:
            self.unsqueeze = None

        # Make post-net data pipes
        # remove "channel" dimensions if neccessary
        self.postnet_pipe = type('SubDataPipe', (object,), {})
        self.postnet_pipe.nocycle = self.scaletanh2img_real + self.scaletanh2img_fake
        self.postnet_pipe.cycle = self.scaletanh2img_real + self.scaletanh2img_fake + self.scaletanh2img_cycled
        if self.ndims == len(self.common_voxel_size):
            self.postnet_pipe.nocycle += gp.Squeeze([self.real, 
                                            self.fake, 
                                            ], axis=1) # remove channel dimension for grayscale
            self.postnet_pipe.cycle += gp.Squeeze([self.real, 
                                            self.fake, 
                                            self.cycled,
                                            ], axis=1) # remove channel dimension for grayscale
        
        # Make training datapipe
        self.train_pipe = self.source + gp.RandomLocation()
        if self.reject:
            self.train_pipe += self.reject
        if self.resample:
            self.train_pipe += self.resample
        self.train_pipe += self.augment
        if self.unsqueeze:
            self.train_pipe += self.unsqueeze # add "channel" dimensions if neccessary, else use z dimension as channel
        # self.train_pipe += gp.Stack(src.batch_size)# add "batch" dimensions    #TODO: ADD IN TRAINING PIP


        # Make predicting datapipe
        self.predict_pipe = self.source
        if self.reject:
            self.predict_pipe += self.reject
        if self.resample:
            self.predict_pipe += self.resample
        self.predict_pipe += self.normalize_real + self.scaleimg2tanh_real
        if self.unsqueeze:
            self.predict_pipe += self.unsqueeze # add "channel" dimensions if neccessary, else use z dimension as channel
        self.predict_pipe += gp.Stack(1)# add "batch" dimensions    
