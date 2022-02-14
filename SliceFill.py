# !conda activate n2v
import itertools
from matplotlib import pyplot as plt
import torch
import glob
import re
import zarr
import daisy
import os

# import gunpowder as gp
import sys
sys.path.insert(0, '/n/groups/htem/users/jlr54/gunpowder/')
from gunpowder import gunpowder as gp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN', 'INFO')

import math
import functools
from tqdm import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True

# from funlib.learn.torch.models.unet import UNet, ConvPass
from residual_unet import ResidualUNet
from unet import UNet, ConvPass
from tri_utils import *
from SliceFill_Model import *
from SliceFill_LossFunctions import *
from SliceFill_Optimizers import *

class SliceFill(): #TODO: Just pass config file or dictionary
    def __init__(self,
            src_path, #EXPECTS ZARR VOLUME
            voxel_size, # voxel size of src (for each dimension)
            src_name='raw',
            mask_name=None, # expects mask to be in same place as real zarr
            out_path=None,
            model_name='SliceFill', #TODO: add in model particulars (gan_mode / generator architecture / etc.)
            model_path='./models/',
            
            gnet_type='resnet',
            gnet_kwargs={
                'input_nc': 1,
                'output_nc': 1,
                # 'num_downs': 3, # unet specific
                'ngf': 64,
                'n_blocks': 9, # resnet specific
            },
            ### below temporarily kept for backward compatibility
            gnet_depth=3, # number of layers in unets (i.e. generators)
            g_downsample_factor=2,
            g_num_fmaps=16,
            g_fmap_inc_factor=2,
            g_constant_upsample=True,            
            g_kernel_size_down=None,
            g_kernel_size_up=None,
            gnet_activation='ReLU',
            residual_unet=False,
            residual_blocks=False,
            padding_unet='same',
            ###
            g_init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004

            dnet_depth=3, # number of layers in Discriminator networks
            d_downsample_factor=2,
            d_num_fmaps=16,
            d_kernel_size=3, 
            d_init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            
            l1_lambda=10, # Default from CycleGAN paper
            loss_style='cycle', # supports 'cycle' or 'split'
            gan_mode='lsgan',
            adam_betas = [0.9, 0.999],
            
            min_coefvar=0,
            side_length=32,
            batch_size=1,
            num_workers=11,
            cache_size=50,
            spawn_subprocess=False,
            num_epochs=10000,
            log_every=100,
            save_every=2000,
            tensorboard_path='./tensorboard/',
            verbose=True,
            checkpoint=None, # Used for prediction/rendering, training always starts from latest            
            ):

            self.src_path = src_path
            self.voxel_size = gp.Coordinate(voxel_size)
            self.src_name = src_name
            self.mask_name = mask_name
            if out_path is None:
                self.out_path = self.src_path
            else:
                self.out_path = out_path
            self.model_name = model_name
            self.model_path = model_path
            self.side_length = side_length

            self.gnet_type = gnet_type.lower()
            self.gnet_kwargs = gnet_kwargs

            self.gnet_depth = gnet_depth
            self.dnet_depth = dnet_depth
            self.g_downsample_factor = g_downsample_factor
            self.d_downsample_factor = d_downsample_factor
            self.g_num_fmaps = g_num_fmaps
            self.d_num_fmaps = d_num_fmaps
            self.g_fmap_inc_factor = g_fmap_inc_factor
            self.g_constant_upsample = g_constant_upsample
            self.g_kernel_size_down=g_kernel_size_down
            self.g_kernel_size_up=g_kernel_size_up
            self.d_kernel_size = d_kernel_size 
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.cache_size = cache_size
            self.num_workers = num_workers
            self.spawn_subprocess = spawn_subprocess            
            self.g_init_learning_rate = g_init_learning_rate
            self.d_init_learning_rate = d_init_learning_rate
            self.l1_lambda = l1_lambda
            self.log_every = log_every
            self.save_every = save_every
            self.tensorboard_path = tensorboard_path
            self.verbose = verbose
            self._set_verbose()
            if checkpoint is None:
                try:
                    self.checkpoint, self.iteration = self._get_latest_checkpoint()
                except:
                    logger.info('Checkpoint not found. Starting from scratch.')
                    self.checkpoint = None
            else:
                self.checkpoint = checkpoint
            self.loss_style = loss_style
            self.adam_betas = adam_betas
            self.min_coefvar = min_coefvar
            self.gnet_activation = gnet_activation
            self.residual_unet = residual_unet
            self.residual_blocks = residual_blocks
            self.padding_unet = padding_unet
            self.gan_mode=gan_mode
            self.build_machine()

    def set_device(self, id=0):
        self.device_id = id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
    
    def set_verbose(self, verbose=True):
        self.verbose = verbose
        self._set_verbose()

    def _set_verbose(self):
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def batch_show(self, batch=None, i=0):
        if batch is None:
            batch = self.batch
        if not hasattr(self, 'col_dict'): 
            self.col_dict = {'REAL':0, 'FAKE':1, 'CYCL':2}#, 'MASK':3}
        rows = (self.real_A in batch.arrays) + (self.real_B in batch.arrays)        
        fig, axes = plt.subplots(rows, len(self.col_dict), figsize=(10*len(self.col_dict), 10*rows))
        for array, value in batch.items():
            label = array.identifier
            c = self.col_dict[label[:4]]
            r = (int('_B' in label) + int('FAKE' in label)) % 2
            if len(value.data.shape) > 3: # pick one from the batch
                img = value.data[i].squeeze()
            else:
                img = value.data.squeeze()
            if len(img.shape) == 3:
                mid = img.shape[0] // 2 # for 3D volume
                data = img[mid]
            else:
                data = img
            if rows == 1:
                axes[c].imshow(data, cmap='gray', vmin=0, vmax=1)
                axes[c].set_title(label)                
            else:
                axes[r, c].imshow(data, cmap='gray', vmin=0, vmax=1)
                axes[r, c].set_title(label)

    def batch_tBoard_write(self, i=0):
        self.trainer.summary_writer.flush()
        self.n_iter = self.trainer.iteration

    def _get_latest_checkpoint(self):
        basename = self.model_path + self.model_name
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        checkpoints = glob.glob(basename + '_checkpoint_*')
        checkpoints.sort(key=natural_keys)

        if len(checkpoints) > 0:

            checkpoint = checkpoints[-1]
            iteration = int(checkpoint.split('_')[-1])
            return checkpoint, iteration

        return None, 0
        
    def get_extents(self, side_length=None, array_name=None):
        if side_length is None:
            side_length = self.side_length

        if (self.padding_unet is not None) and (self.padding_unet.lower() == 'valid'):
            if array_name is not None and 'pred' in array_name.lower():
                shape = (1,2) + (side_length,) * 2 # simulate adj_slices input to generator
                result = self.Gnet(torch.rand(*shape))
                side_length = result.shape[-1]
                
        extents = np.ones((len(self.voxel_size)))
        if array_name.lower() == 'real':
            extents *= 3 # get 3 slices
        extents[-2:] = side_length # assumes first dimension is z (i.e. the dimension slices are stacked in)
        return gp.Coordinate(extents)

    def get_request(self, side_length=None):
        request = gp.BatchRequest()
        for array in self.arrays:                        
            if array.identifier.lower() == 'adj_slices':
                request.add(array, None, self.voxel_size) # get 2 slices on either side of middle   
            else:
                extents = self.get_extents(side_length, array_name=array.identifier)
                request.add(array, self.voxel_size * extents, self.voxel_size)
        
        return request

    def get_generator(self): 
        if self.gnet_type == 'unet':

            generator = UnetGenerator(**self.gnet_kwargs)
            
            # if self.residual_unet:
            #     unet = ResidualUNet(
            #             in_channels=2,
            #             num_fmaps=self.g_num_fmaps,
            #             fmap_inc_factor=self.g_fmap_inc_factor,
            #             downsample_factors=[(self.g_downsample_factor,)*2,] * (self.gnet_depth - 1),
            #             padding=self.padding_unet,
            #             constant_upsample=self.g_constant_upsample,
            #             voxel_size=self.voxel_size[-2:],
            #             kernel_size_down=self.g_kernel_size_down,
            #             kernel_size_up=self.g_kernel_size_up,
            #             residual=self.residual_blocks,
            #             activation=self.gnet_activation
            #             )
            #     generator = torch.nn.Sequential(
            #                         unet, 
            #                         torch.nn.Tanh())
            # else:
            #     unet = UNet(
            #             in_channels=2,
            #             num_fmaps=self.g_num_fmaps,
            #             fmap_inc_factor=self.g_fmap_inc_factor,
            #             downsample_factors=[(self.g_downsample_factor,)*2,] * (self.gnet_depth - 1),
            #             padding=self.padding_unet,
            #             constant_upsample=self.g_constant_upsample,
            #             voxel_size=self.common_voxel_size[-2:],
            #             kernel_size_down=self.g_kernel_size_down,
            #             kernel_size_up=self.g_kernel_size_up,
            #             residual=self.residual_blocks,
            #             activation=self.gnet_activation
            #             )
            #     generator = torch.nn.Sequential(
            #                         unet,
            #                         ConvPass(self.g_num_fmaps, 1, [(1,)*2], activation=None, padding=self.padding_unet), 
            #                         torch.nn.Tanh())
            
        elif self.gnet_type == 'resnet':
            
            generator = ResnetGenerator(**self.gnet_kwargs)            

        else:

            raise f'Unknown generator type requested: {self.gnet_type}'

        if self.gnet_activation is not None:
            init_weights(generator, init_type='kaiming', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE
        else:
            init_weights(generator, init_type='normal', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE
        return generator

    def get_discriminator(self):
        norm_instance = torch.nn.InstanceNorm2d
        discriminator_maker = NLayerDiscriminator

        norm_layer = functools.partial(norm_instance, affine=False, track_running_stats=False)
        discriminator = discriminator_maker(input_nc=3, 
                                        ndf=self.d_num_fmaps, 
                                        n_layers=self.dnet_depth, 
                                        norm_layer=norm_layer,
                                        downsampling_kw=self.d_downsample_factor, 
                                        kw=self.d_kernel_size,
                                 )
                                 
        init_weights(discriminator, init_type='kaiming')
        return discriminator

    def setup_networks(self):
        self.Gnet = self.get_generator()
        self.Dnet = self.get_discriminator()

    def setup_model(self):
        if not hasattr(self, 'Gnet'):
            self.setup_networks()        

        self.model = SliceFill_Model(self.Gnet)
       
        if self.loss_style.lower()=='l1' or self.loss_style.lower()=='care':
            
            self.optimizer_G = torch.optim.Adam(self.Gnet.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)
            self.optimizer = SliceFill_CARE_Optimizer(self.optimizer_G)
            
            self.loss = SliceFill_CARE_Loss(self.Gnet, self.optimizer_G)
        
        elif self.loss_style.lower()=='conditionalgan':
        
            self.optimizer_G = torch.optim.Adam(self.Gnet.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)
            self.optimizer_D = torch.optim.Adam(self.Dnet.parameters(), lr=self.d_init_learning_rate, betas=self.adam_betas)
            self.optimizer = SliceFill_ConditionalGAN_Optimizer(self.optimizer_G, self.optimizer_D)

            self.loss = SliceFill_ConditionalGAN_Loss(self.Gnet, self.Dnet, self.optimizer_G, self.optimizer_D, l1_lambda=self.l1_lambda, gan_mode=self.gan_mode)

        else:

            print("Unexpected Loss Style. Accepted options are 'l1' (a.k.a. 'care') or 'conditionalgan'")
            raise

    def build_machine(self):       
        # initialize needed variables
        self.arrays = []

        # define our network model for training
        self.setup_networks()        
        self.setup_model()

        # get performance stats
        self.performance = gp.PrintProfilingStats(every=self.log_every)

        # setup a cache
        self.cache = gp.PreCache(num_workers=self.num_workers, cache_size=self.cache_size)

        # declare arrays to use in the pipelines
        array_names = [ 'real',
                        'adj_slices',
                        'real_mid_slice',
                        'pred_mid_slice']
        if self.mask_name is not None: 
            array_names += ['mask']
            self.masked = True
        else:
            self.masked = False
                
        for array_name in array_names:            
            array_key = gp.ArrayKey(array_name.upper())
            setattr(self, array_name, array_key) # add ArrayKeys to object
            self.arrays += [array_key]
            if 'slice' in array_name:            
                setattr(self, 'normalize_'+array_name, gp.Normalize(array_key)) # add normalizations, if appropriate                       

        # setup data sources
        self.src_names = {self.real: self.src_name}
        self.src_specs = {self.real: gp.ArraySpec(interpolatable=True, voxel_size=self.voxel_size)}
        if self.masked: 
            self.src_names[self.mask] = self.mask_name
            self.src_specs[self.mask] = gp.ArraySpec(interpolatable=False)
        self.source = gp.ZarrSource(    # add the data source
                    self.src_path,  
                    self.src_names,  # which dataset to associate to the array key
                    self.src_specs  # meta-information
        )
        
        # setup rejections
        self.reject = None
        if self.masked:
            self.reject = gp.Reject(mask = self.mask, min_masked=0.999)

        if self.min_coefvar is not False:
            if self.reject is None:
                self.reject = gp.RejectConstant(self.real, min_coefvar = self.min_coefvar, axis = 1)
            else:
                self.reject += gp.RejectConstant(self.real, min_coefvar = self.min_coefvar, axis = 1)
        
        self.build_train_parts()

    def build_train_parts(self):
        # define axes for mirroring and transpositions
        self.augment_axes = list(np.arange(3)[-2:])
        self.augment = gp.SimpleAugment(mirror_only = self.augment_axes, transpose_only = self.augment_axes)
        self.augment += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
                    control_point_spacing=self.side_length//2,
                    jitter_sigma=(5.0, 5.0,),
                    rotation_interval=(0, math.pi/2),
                    subsample=4,
                    spatial_dims=2
        )

        # create a train node using our model, loss, and optimizer
        self.trainer = gp.torch.Train(
                            self.model,
                            self.loss,
                            self.optimizer,
                            inputs = {
                                'input': self.real,
                            },
                            outputs = {
                                0: self.adj_slices,
                                1: self.real_mid_slice,
                                2: self.pred_mid_slice
                            },
                            loss_inputs = {
                                0: self.adj_slices,
                                1: self.real_mid_slice,
                                2: self.pred_mid_slice
                            },
                            log_dir=self.tensorboard_path,
                            log_every=self.log_every,
                            checkpoint_basename=self.model_path+self.model_name,
                            save_every=self.save_every,
                            spawn_subprocess=self.spawn_subprocess
                            )

        # Make training pipeline
        self.train_pipe = self.source + gp.RandomLocation()
        if self.reject:
            self.train_pipe += self.reject
        self.train_pipe += self.augment + gp.Stack(self.batch_size) + self.trainer
        
        # remove "channel" dimension if neccessary
        self.train_pipe += gp.Squeeze([ self.real_mid_slice, 
                                        self.pred_mid_slice
                                        ], axis=1) # remove channel dimension for grayscale
        if self.batch_size == 1:
            self.train_pipe += gp.Squeeze([self.real, 
                                            self.adj_slices, 
                                            self.real_mid_slice, 
                                            self.pred_mid_slice
                                            ], axis=0)
        self.train_pipe += self.normalize_adj_slices + self.normalize_real_mid_slice + self.normalize_pred_mid_slice
        self.test_train_pipe = self.train_pipe.copy() + self.performance
        self.train_pipe += self.cache

        # create request
        self.train_request = self.get_request()
            
    def test_train(self):
        if self.test_train_pipe is None:
            self.build_training_pipeline()
        self.model.train()
        with gp.build(self.test_train_pipe):
            self.batch = self.test_train_pipe.request_batch(self.train_request)
        self.batch_show()
        return self.batch

    def train(self):
        if self.train_pipe is None:
            self.build_machine()
        self.model.train()
        with gp.build(self.train_pipe):
            for i in tqdm(range(self.num_epochs)):
                self.batch = self.train_pipe.request_batch(self.train_request)
                if hasattr(self.loss, 'loss_dict'):
                    print(self.loss.loss_dict)
                if i % self.log_every == 0:
                    self.batch_tBoard_write()
        return self.batch
        
    def test_prediction(self, side_length=None):
        #set model into evaluation mode
        self.model.eval()
        # model_outputs = {
        #     0: self.adj_slices,
        #     1: self.real_mid_slice,
        #     2: self.pred_mid_slice
        # }        
        
        self.predict_pipe = self.source + gp.RandomLocation() 
        if self.reject: self.predict_pipe += self.reject

        self.predict_pipe += gp.Unsqueeze([self.real]) # add batch dimension
        self.predict_pipe += gp.torch.Predict(self.model,
                                inputs = {'input': self.real},
                                outputs = {
                                    0: self.adj_slices,
                                    1: self.real_mid_slice,
                                    2: self.pred_mid_slice
                                },
                                checkpoint = self.checkpoint
                                )
        
        # remove "channel" dimension if neccessary
        self.predict_pipe += gp.Squeeze([ self.real_mid_slice, 
                                        self.pred_mid_slice
                                        ], axis=1) 
        # remove "batch" dimension
        self.predict_pipe += gp.Squeeze([self.real, 
                                        self.adj_slices, 
                                        self.real_mid_slice, 
                                        self.pred_mid_slice
                                        ], axis=0)
        self.predict_pipe += self.normalize_adj_slices + self.normalize_real_mid_slice + self.normalize_pred_mid_slice

        self.pred_request = self.get_request(side_length)

        with gp.build(self.predict_pipe):
            self.batch = self.predict_pipe.request_batch(self.pred_request)

        self.batch_show()
        return self.batch

    def render_full(self, side_length=None):
        #set model into evaluation mode
        self.model.eval()
        # model_outputs = {
        #     0: self.adj_slices,
        #     1: self.real_mid_slice,
        #     2: self.pred_mid_slice
        # }        
    
        # set prediction spec 
        if self.source.spec is None:
            data_file = zarr.open(self.src_path)
            pred_spec = self.source._Hdf5LikeSource__read_spec(self.real, data_file, self.src_name).copy()
        else:
            pred_spec = self.source.spec[self.real].copy()        
        pred_spec.voxel_size = self.voxel_size
        pred_spec.dtype = self.normalize_pred_mid_slice.dtype

        scan_request = self.get_request(side_length)

        #Declare new array to write to
        if not hasattr(self, 'compressor'):
            self.compressor = {'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': 64
                }
        
        dataset_names = {self.pred_mid_slice: 'volumes/'+self.model_name+'_enFAKE'}
        array_specs = {self.pred_mid_slice: pred_spec.copy()}

        source_ds = daisy.open_ds(self.src_path, self.src_name)
        self.total_roi = source_ds.data_roi
        for key, name in dataset_names.items():
            write_size = scan_request[key].roi.get_shape()
            daisy.prepare_ds(
                self.out_path,
                name,
                self.total_roi,
                daisy.Coordinate(self.common_voxel_size),
                np.uint8,
                write_size=write_size,
                num_channels=1,
                compressor=self.compressor)

        self.render_pipe = self.source 
        self.render_pipe += gp.Unsqueeze([self.real]) # add batch dimension

        self.render_pipe += gp.torch.Predict(self.model,
                                inputs = {'input': self.real},
                                outputs = {
                                    # 0: self.adj_slices,
                                    # 1: self.real_mid_slice,
                                    2: self.pred_mid_slice
                                },
                                checkpoint = self.checkpoint,
                                array_specs = array_specs, 
                                spawn_subprocess=self.spawn_subprocess
                                )
        
        # remove "channel" dimension if neccessary
        self.render_pipe += gp.Squeeze([ 
                                        # self.real_mid_slice, 
                                        self.pred_mid_slice
                                        ], axis=1) 
        # remove "batch" dimension
        self.render_pipe += gp.Squeeze([
                                        # self.real, 
                                        # self.adj_slices, 
                                        # self.real_mid_slice, 
                                        self.pred_mid_slice
                                        ], axis=0)
        self.render_pipe += self.normalize_pred_mid_slice # + self.normalize_adj_slices + self.normalize_real_mid_slice
        
        self.render_pipe += gp.AsType(self.pred_mid_slice, np.uint8)       

        self.render_pipe += gp.ZarrWrite(
                        dataset_names = dataset_names,
                        output_filename = self.out_path,
                        compression_type = self.compressor
                        # dataset_dtypes = {fake: pred_spec.dtype}
                        )        
        
        self.render_pipe += gp.Scan(scan_request, num_workers=self.num_workers, cache_size=self.cache_size)

        request = gp.BatchRequest()

        print(f'Full rendering pipeline declared. Building...')
        with gp.build(self.render_pipe):
            print('Starting full volume render...')
            self.render_pipe.request_batch(request)
            print('Finished.')

    def load_saved_model(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = self.checkpoint
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict()
        else:
            raise('No saved checkpoint found.')
