# !conda activate n2v
import itertools
from matplotlib import pyplot as plt
import torch
import glob
import re
import zarr
import daisy
import os

import gunpowder as gp
# import sys
# sys.path.insert(0, '/n/groups/htem/users/jlr54/gunpowder/')
# from gunpowder import gunpowder as gp

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
try:
    from .CycleGAN_Model import *
    from .CycleGAN_LossFunctions import *
    from .CycleGAN_Optimizers import *
except:
    from CycleGAN_Model import *
    from CycleGAN_LossFunctions import *
    from CycleGAN_Optimizers import *
class CycleGAN(): #TODO: Just pass config file or dictionary
    def __init__(self,
            src_A, #EXPECTS ZARR VOLUME
            src_B,
            A_voxel_size, # voxel size of src_A (for each dimension)
            B_voxel_size, # voxel size of src_B (for each dimension)
            common_voxel_size=None, # voxel size to resample A and B into for training
            ndims=None,
            A_name='raw',
            B_name='raw',
            mask_A_name=None, # expects mask to be in same place as real zarr
            mask_B_name=None,
            A_out_path=None,
            B_out_path=None,
            model_name='CycleGun',
            model_path='./models/',
            
            gnet_type='unet',
            gnet_kwargs = {},
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
            identity_lambda=0.5, # Default from CycleGAN paper
            loss_style='cycle', # supports 'cycle' or 'split'
            gan_mode='lsgan',
            sampling_bottleneck=False,
            adam_betas = [0.9, 0.999],
            
            min_coefvar=None,
            interp_order=None,
            side_length=32,# in dataset A sized voxels at output layer - actual used ROI for network input will be bigger for valid padding
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

            self.src_A = src_A
            self.src_B = src_B
            self.A_voxel_size = gp.Coordinate(A_voxel_size)
            self.B_voxel_size = gp.Coordinate(B_voxel_size)
            if common_voxel_size is None:
                self.common_voxel_size = self.B_voxel_size
            else:
                self.common_voxel_size = gp.Coordinate(common_voxel_size)
            if ndims is None:
                self.ndims = sum(np.array(self.common_voxel_size) == np.min(self.common_voxel_size))            
            else:
                self.ndims = ndims
            self.A_name = A_name
            self.B_name = B_name
            self.mask_A_name = mask_A_name
            self.mask_B_name = mask_B_name
            if A_out_path is None:
                self.A_out_path = self.src_A
            else:
                self.A_out_path = A_out_path
            if B_out_path is None:
                self.B_out_path = self.src_B
            else:
                self.B_out_path = B_out_path
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
            self.identity_lambda = identity_lambda
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
            self.interp_order = interp_order
            self.loss_style = loss_style
            self.sampling_bottleneck =sampling_bottleneck
            self.adam_betas = adam_betas
            self.min_coefvar = min_coefvar
            self.gnet_activation = gnet_activation
            self.residual_unet = residual_unet
            self.residual_blocks = residual_blocks
            self.padding_unet = padding_unet
            self.gan_mode=gan_mode
            self.build_machine()
            self.training_pipeline = None
            self.test_training_pipeline = None

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

    def batch_show(self, batch=None, i=0, show_mask=False):
        if batch is None:
            batch = self.batch
        if not hasattr(self, 'col_dict'): 
            self.col_dict = {'REAL':0, 'FAKE':1, 'CYCL':2}
        if show_mask: self.col_dict['MASK'] = 3
        rows = (self.real_A in batch.arrays) + (self.real_B in batch.arrays)       
        cols = 0
        for key in self.col_dict.keys():
            cols += key in [array.identifier[:4] for array in batch.arrays] 
        fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 10*rows))
        for array, value in batch.items():
            label = array.identifier
            if label[:4] in self.col_dict:
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

    def write_tBoard_graph(self, batch=None):
        if batch is None:
            batch = self.batch
        
        ex_inputs = []
        if self.real_A in batch:
            ex_inputs += [torch.tensor(batch[self.real_A].data)]
        if self.real_B in batch:
            ex_inputs += [torch.tensor(batch[self.real_B].data)]

        for i, ex_input in enumerate(ex_inputs):
            if self.ndims == len(self.common_voxel_size): # add channel dimension if necessary
                ex_input = ex_input.unsqueeze(axis=1)
            if self.batch_size == 1: # ensure batch dimension is present
                ex_input = ex_input.unsqueeze(axis=0)
            ex_inputs[i] = ex_input
        
        self.trainer.summary_writer.add_graph(self.model, ex_inputs)                

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
            if array_name is not None and not ('real' in array_name.lower() or 'mask' in array_name.lower()):
                shape = (1,1) + (side_length,) * self.ndims
                result = self.netG1(torch.rand(*shape))
                if 'fake' in array_name.lower():
                    side_length = result.shape[-1]
                elif 'cycle' in array_name.lower():
                    result = self.netG1(result)
                    side_length = result.shape[-1]

        extents = np.ones((len(self.common_voxel_size)))
        extents[-self.ndims:] = side_length # assumes first dimension is z (i.e. the dimension breaking isotropy)
        return gp.Coordinate(extents)

    def get_generator(self): 
        if self.gnet_type == 'unet':

            # if self.ndims == 2:
            #     generator = UnetGenerator(**self.gnet_kwargs)
            
            # elif self.ndims == 3:
            #     generator = UnetGenerator3D(**self.gnet_kwargs)

            # else:
            #     raise f'Unet generators only specified for 2D or 3D, not {self.ndims}D'

            if self.residual_unet:
                unet = ResidualUNet(
                        in_channels=1,
                        num_fmaps=self.g_num_fmaps,
                        fmap_inc_factor=self.g_fmap_inc_factor,
                        downsample_factors=[(self.g_downsample_factor,)*self.ndims,] * (self.gnet_depth - 1),
                        padding=self.padding_unet,
                        constant_upsample=self.g_constant_upsample,
                        voxel_size=self.common_voxel_size[-self.ndims:],
                        kernel_size_down=self.g_kernel_size_down,
                        kernel_size_up=self.g_kernel_size_up,
                        residual=self.residual_blocks,
                        activation=self.gnet_activation
                        )
                generator = torch.nn.Sequential(
                                    unet, 
                                    torch.nn.Tanh()
                                    )
            else:
                unet = UNet(
                        in_channels=1,
                        num_fmaps=self.g_num_fmaps,
                        fmap_inc_factor=self.g_fmap_inc_factor,
                        downsample_factors=[(self.g_downsample_factor,)*self.ndims,] * (self.gnet_depth - 1),
                        padding=self.padding_unet,
                        constant_upsample=self.g_constant_upsample,
                        voxel_size=self.common_voxel_size[-self.ndims:],
                        kernel_size_down=self.g_kernel_size_down,
                        kernel_size_up=self.g_kernel_size_up,
                        residual=self.residual_blocks,
                        activation=self.gnet_activation
                        )
                generator = torch.nn.Sequential(
                                    unet,
                                    ConvPass(self.g_num_fmaps, 1, [(1,)*self.ndims], activation=None, padding=self.padding_unet), 
                                    torch.nn.Tanh()
                                    )
            
        elif self.gnet_type == 'resnet':
            
            if self.ndims == 2:
                generator = ResnetGenerator(**self.gnet_kwargs)
            
            elif self.ndims == 3:
                generator = ResnetGenerator3D(**self.gnet_kwargs)

            else:
                raise f'Resnet generators only specified for 2D or 3D, not {self.ndims}D'

        else:

            raise f'Unknown generator type requested: {self.gnet_type}'

        if self.gnet_activation is not None:
            init_weights(generator, init_type='kaiming', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE
        else:
            init_weights(generator, init_type='normal', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE
        return generator

    def get_discriminator(self, conf=None):
        if conf is None: conf = self
        if conf.ndims == 3: #3D case
            norm_instance = torch.nn.InstanceNorm3d
            discriminator_maker = NLayerDiscriminator3D
        elif conf.ndims == 2:
            norm_instance = torch.nn.InstanceNorm2d
            discriminator_maker = NLayerDiscriminator

        norm_layer = functools.partial(norm_instance, affine=False, track_running_stats=False)
        discriminator = discriminator_maker(input_nc=1, 
                                        ndf=conf.d_num_fmaps, 
                                        n_layers=conf.dnet_depth, 
                                        norm_layer=norm_layer,
                                        downsampling_kw=conf.d_downsample_factor, 
                                        kw=conf.d_kernel_size,
                                 )
                                 
        init_weights(discriminator, init_type='kaiming')
        return discriminator

    def setup_networks(self):
        self.netG1 = self.get_generator()
        self.netG2 = self.get_generator()
        
        self.netD1 = self.get_discriminator()
        self.netD2 = self.get_discriminator()

    def setup_model(self):
        if not hasattr(self, 'netG1'):
            self.setup_networks()

        if self.sampling_bottleneck:
            # scale_factor_A = (1,1) + tuple(np.divide(self.common_voxel_size, self.A_voxel_size)[-self.ndims:])
            scale_factor_A = tuple(np.divide(self.common_voxel_size, self.A_voxel_size)[-self.ndims:])
            if not any([s < 1 for s in scale_factor_A]): scale_factor_A = None
            # scale_factor_B = (1,1) + tuple(np.divide(self.common_voxel_size, self.B_voxel_size)[-self.ndims:])
            scale_factor_B = tuple(np.divide(self.common_voxel_size, self.B_voxel_size)[-self.ndims:])
            if not any([s < 1 for s in scale_factor_B]): scale_factor_B = None
        else:
            scale_factor_A, scale_factor_B = None, None

        if self.padding_unet.lower() == 'valid':
            padding = 'valid'
        else:
            padding = None
        # self.l1_loss = torch.nn.SmoothL1Loss() 
        self.l1_loss = torch.nn.L1Loss() 
        self.gan_loss = GANLoss(gan_mode=self.gan_mode)
        if self.loss_style.lower()=='cycle':
            
            self.model = CycleGAN_Model(self.netG1, self.netD1, self.netG2, self.netD2, scale_factor_A, scale_factor_B)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()), lr=self.g_init_learning_rate, betas=self.adam_betas)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), lr=self.d_init_learning_rate, betas=self.adam_betas)
            self.optimizer = CycleGAN_Optimizer(self.optimizer_G, self.optimizer_D)
            
            self.loss = CycleGAN_Loss(self.l1_loss, self.gan_loss, self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_D, self.optimizer_G, self.ndims, self.l1_lambda, self.identity_lambda, padding, self.gan_mode)
        
        elif self.loss_style.lower()=='split':
        
            self.model = CycleGAN_Split_Model(self.netG1, self.netD1, self.netG2, self.netD2, scale_factor_A, scale_factor_B)
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), lr=self.d_init_learning_rate, betas=self.adam_betas)
            self.optimizer = Split_CycleGAN_Optimizer(self.optimizer_G1, self.optimizer_G2, self.optimizer_D)

            self.loss = SplitGAN_Loss(self.l1_loss, self.gan_loss, self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_G1, self.optimizer_G2, self.optimizer_D, self.ndims, self.l1_lambda, self.identity_lambda, padding, self.gan_mode)

        else:

            print("Unexpected Loss Style. Accepted options are 'cycle' or 'split'")
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
        self.cache = gp.PreCache(num_workers=self.num_workers, cache_size=self.cache_size)#os.cpu_count())

        # define axes for mirroring and transpositions
        self.augment_axes = list(np.arange(3)[-self.ndims:])

        # build datapipes
        self.datapipe_A = self.get_datapipe('A')
        self.datapipe_B = self.get_datapipe('B') #datapipe has: train_pipe, source, reject, resample, augment, unsqueeze, etc.}

    def get_datapipe(self, side):
        datapipe = type('DataPipe', (object,), {}) # make simple object to smoothly store variables
        side = side.upper() # ensure uppercase
        datapipe.src_voxel_size = getattr(self, side+'_voxel_size')
        
        # declare arrays to use in the pipelines
        array_names = ['real', 
                        'fake', 
                        'cycled']
        if getattr(self, f'mask_{side}_name') is not None: 
            array_names += ['mask']
            datapipe.masked = True
        else:
            datapipe.masked = False
                
        for array in array_names:
            if 'fake' in array:
                other_side = ['A','B']
                other_side.remove(side)
                array_name = array + '_' + other_side[0]
            else:
                array_name = array + '_' + side
            array_key = gp.ArrayKey(array_name.upper())
            setattr(datapipe, array, array_key) # add ArrayKeys to object
            setattr(self, array_name, array_key) 
            self.arrays += [array_key]
            #add normalizations and scaling, if appropriate        
            if 'mask' not in array:            
                setattr(datapipe, 'scaletanh2img_'+array, gp.IntensityScaleShift(array_key, 0.5, 0.5))
                setattr(self, 'scaletanh2img_'+array_name, gp.IntensityScaleShift(array_key, 0.5, 0.5))            
                
                if 'real' in array:                        
                    setattr(datapipe, 'normalize_'+array, gp.Normalize(array_key))
                    setattr(self, 'normalize_'+array_name, gp.Normalize(array_key))
                    setattr(datapipe, 'scaleimg2tanh_'+array, gp.IntensityScaleShift(array_key, 2, -1))
                    setattr(self, 'scaleimg2tanh_'+array_name, gp.IntensityScaleShift(array_key, 2, -1))
        
        #Setup sources and resampling nodes
        if self.common_voxel_size != datapipe.src_voxel_size:
            datapipe.real_src = gp.ArrayKey(f'REAL_{side}_SRC')
            setattr(self, f'real_{side}_src', datapipe.real_src)
            datapipe.resample = gp.Resample(datapipe.real_src, self.common_voxel_size, datapipe.real, ndim=self.ndims, interp_order=self.interp_order)
            if datapipe.masked: 
                datapipe.mask_src = gp.ArrayKey(f'MASK_{side}_SRC')
                setattr(self, f'mask_{side}_src', datapipe.mask_src)
                datapipe.resample += gp.Resample(datapipe.mask_src, self.common_voxel_size, datapipe.mask, ndim=self.ndims, interp_order=self.interp_order)
        else:            
            datapipe.real_src = datapipe.real
            datapipe.resample = None
            if datapipe.masked: 
                datapipe.mask_src = datapipe.mask

        # setup data sources
        datapipe.src_path = getattr(self, 'src_'+side)# the zarr container
        datapipe.out_path = getattr(self, side+'_out_path')
        datapipe.real_name = getattr(self, side+'_name')
        datapipe.src_names = {datapipe.real_src: datapipe.real_name}
        datapipe.src_specs = {datapipe.real_src: gp.ArraySpec(interpolatable=True, voxel_size=datapipe.src_voxel_size)}
        if datapipe.masked: 
            datapipe.mask_name = getattr(self, f'mask_{side}_name')
            datapipe.src_names[datapipe.mask_src] = datapipe.mask_name
            datapipe.src_specs[datapipe.mask_src] = gp.ArraySpec(interpolatable=False)
        datapipe.source = gp.ZarrSource(    # add the data source
                    datapipe.src_path,  
                    datapipe.src_names,  # which dataset to associate to the array key
                    datapipe.src_specs  # meta-information
        )
        
        # setup rejections
        datapipe.reject = None
        if datapipe.masked:
            datapipe.reject = gp.Reject(mask = datapipe.mask_src, min_masked=0.999)

        if self.min_coefvar:
            if datapipe.reject is None:
                datapipe.reject = gp.RejectConstant(datapipe.real_src, min_coefvar = self.min_coefvar)
            else:
                datapipe.reject += gp.RejectConstant(datapipe.real_src, min_coefvar = self.min_coefvar)

        datapipe.augment = gp.SimpleAugment(mirror_only = self.augment_axes, transpose_only = self.augment_axes)
        datapipe.augment += datapipe.normalize_real
        datapipe.augment += datapipe.scaleimg2tanh_real
        datapipe.augment += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
                    control_point_spacing=100//self.common_voxel_size[-self.ndims], # self.side_length//2,
                    # jitter_sigma=(5.0,)*self.ndims,
                    jitter_sigma=(0., 5.0, 5.0,)[-self.ndims:],
                    rotation_interval=(0, math.pi/2),
                    subsample=4,
                    spatial_dims=self.ndims
        )

        # add "channel" dimensions if neccessary, else use z dimension as channel
        if self.ndims == len(self.common_voxel_size):
            datapipe.unsqueeze = gp.Unsqueeze([datapipe.real])
        else:
            datapipe.unsqueeze = None

        # Make post-net data pipes
        # remove "channel" dimensions if neccessary
        datapipe.postnet_pipe = type('SubDataPipe', (object,), {})
        datapipe.postnet_pipe.nocycle = datapipe.scaletanh2img_real + datapipe.scaletanh2img_fake
        datapipe.postnet_pipe.cycle = datapipe.scaletanh2img_real + datapipe.scaletanh2img_fake + datapipe.scaletanh2img_cycled
        if self.ndims == len(self.common_voxel_size):
            datapipe.postnet_pipe.nocycle += gp.Squeeze([datapipe.real, 
                                            datapipe.fake, 
                                            ], axis=1) # remove channel dimension for grayscale
            datapipe.postnet_pipe.cycle += gp.Squeeze([datapipe.real, 
                                            datapipe.fake, 
                                            datapipe.cycled,
                                            ], axis=1) # remove channel dimension for grayscale
        
        # Make training datapipe
        datapipe.train_pipe = datapipe.source + gp.RandomLocation()
        if datapipe.reject:
            datapipe.train_pipe += datapipe.reject
        if datapipe.resample:
            datapipe.train_pipe += datapipe.resample
        datapipe.train_pipe += datapipe.augment
        if datapipe.unsqueeze:
            datapipe.train_pipe += datapipe.unsqueeze # add "channel" dimensions if neccessary, else use z dimension as channel
        datapipe.train_pipe += gp.Stack(self.batch_size)# add "batch" dimensions    
        setattr(self, 'pipe_'+side, datapipe.train_pipe)

        return datapipe

    def build_training_pipeline(self):
        # create a train node using our model, loss, and optimizer
        self.trainer = gp.torch.Train(
                            self.model,
                            self.loss,
                            self.optimizer,
                            inputs = {
                                'real_A': self.real_A,
                                'real_B': self.real_B
                            },
                            outputs = {
                                0: self.fake_B,
                                1: self.cycled_B,
                                2: self.fake_A,
                                3: self.cycled_A
                            },
                            loss_inputs = {
                                0: self.real_A,
                                1: self.fake_A,
                                2: self.cycled_A,
                                3: self.real_B,
                                4: self.fake_B,
                                5: self.cycled_B,
                            },
                            log_dir=self.tensorboard_path,
                            log_every=self.log_every,
                            checkpoint_basename=self.model_path+self.model_name,
                            save_every=self.save_every,
                            spawn_subprocess=self.spawn_subprocess
                            )

        # assemble pipeline
        self.training_pipeline = (self.pipe_A, self.pipe_B) + gp.MergeProvider() #merge upstream pipelines for two sources
        self.training_pipeline += self.trainer        
        self.training_pipeline += self.datapipe_A.postnet_pipe.cycle + self.datapipe_B.postnet_pipe.cycle
        if self.batch_size == 1:
            self.training_pipeline += gp.Squeeze([self.real_A, 
                                            self.fake_A, 
                                            self.cycled_A, 
                                            self.real_B, 
                                            self.fake_B, 
                                            self.cycled_B
                                            ], axis=0)
        self.test_training_pipeline = self.training_pipeline.copy() + self.performance
        self.training_pipeline += self.cache

        # create request
        self.train_request = gp.BatchRequest()
        for array in self.arrays:            
            extents = self.get_extents(array_name=array.identifier)
            self.train_request.add(array, self.common_voxel_size * extents, self.common_voxel_size)
            
    def test_train(self):
        if self.test_training_pipeline is None:
            self.build_training_pipeline()
        self.model.train()
        with gp.build(self.test_training_pipeline):
            self.batch = self.test_training_pipeline.request_batch(self.train_request)
        self.batch_show()
        return self.batch

    def train(self):
        if self.training_pipeline is None:
            self.build_training_pipeline()
        self.model.train()
        with gp.build(self.training_pipeline):
            for i in tqdm(range(self.num_epochs)):
                self.batch = self.training_pipeline.request_batch(self.train_request)
                if i == 1:
                    self.write_tBoard_graph()
                if hasattr(self.loss, 'loss_dict'):
                    print(self.loss.loss_dict)
                if i % self.log_every == 0:
                    self.batch_tBoard_write()
        return self.batch
        
    def test_prediction(self, side='A', side_length=None, cycle=True):
        #set model into evaluation mode
        self.model.eval()
        self.model.cycle = cycle
        # model_outputs = {
        #     0: self.fake_B,
        #     1: self.cycled_B,
        #     2: self.fake_A,
        #     3: self.cycled_A}

        #datapipe has: train_pipe, source, reject, resample, augment, unsqueeze, etc.}
        datapipe = getattr(self, 'datapipe_'+side)
        arrays = [datapipe.real, datapipe.fake]
        if cycle:
            arrays += [datapipe.cycled]
        squeeze_arrays = arrays.copy()
        if datapipe.masked:
            arrays += [datapipe.mask]

        input_dict = {'real_'+side: datapipe.real}

        if side=='A':
            output_dict = {0: datapipe.fake}
            if cycle:
                output_dict[3] = datapipe.cycled
        else:        
            output_dict = {2: datapipe.fake}
            if cycle:
                output_dict[1] = datapipe.cycled   

        predict_pipe = datapipe.source + gp.RandomLocation() 
        if datapipe.reject: predict_pipe += datapipe.reject
        if datapipe.resample: predict_pipe += datapipe.resample
        predict_pipe += datapipe.normalize_real
        predict_pipe += datapipe.scaleimg2tanh_real

        if datapipe.unsqueeze: # add "channel" dimensions if neccessary, else use z dimension as channel
            predict_pipe += datapipe.unsqueeze
        predict_pipe += gp.Unsqueeze([datapipe.real]) # add batch dimension

        predict_pipe += gp.torch.Predict(self.model,
                                inputs = input_dict,
                                outputs = output_dict,
                                checkpoint = self.checkpoint
                                )
        
        if cycle:
            predict_pipe += datapipe.postnet_pipe.cycle
        else:
            predict_pipe += datapipe.postnet_pipe.nocycle

        predict_pipe += gp.Squeeze(squeeze_arrays, axis=0) # remove batch dimension

        request = gp.BatchRequest()
        for array in arrays:            
            extents = self.get_extents(side_length, array_name=array.identifier)
            request.add(array, self.common_voxel_size * extents, self.common_voxel_size)

        with gp.build(predict_pipe):
            self.batch = predict_pipe.request_batch(request)

        self.batch_show()
        return self.batch

    def render_full(self, side='A', side_length=None, cycle=False):
        #CYCLED CURRENTLY SAVED IN UPSAMPLED FORM (i.e. not original voxel size)
        #set model into evaluation mode
        self.model.eval()
        self.model.cycle = cycle
        # model_outputs = {
        #     0: self.fake_B,
        #     1: self.cycled_B,
        #     2: self.fake_A,
        #     3: self.cycled_A}
        
        #datapipe has: train_pipe, source, reject, resample, augment, unsqueeze, etc.}
        datapipe = getattr(self, 'datapipe_'+side)        
        arrays = [datapipe.real, datapipe.fake]
        if cycle:
            arrays += [datapipe.cycled]

        input_dict = {'real_'+side: datapipe.real}                

        if side=='A':
            output_dict = {0: datapipe.fake},
            if cycle:
                output_dict[3] = datapipe.cycled
        else:        
            output_dict = {2: datapipe.fake},
            if cycle:
                output_dict[1] = datapipe.cycled   

        # set prediction spec 
        if datapipe.source.spec is None:
            data_file = zarr.open(datapipe.src_path)
            pred_spec = datapipe.source._Hdf5LikeSource__read_spec(datapipe.real, data_file, datapipe.real_name).copy()
        else:
            pred_spec = datapipe.source.spec[datapipe.real].copy()        
        pred_spec.voxel_size = self.common_voxel_size
        pred_spec.dtype = datapipe.normalize_fake.dtype

        dataset_names = {datapipe.fake: 'volumes/'+self.model_name+'_enFAKE'}
        array_specs = {datapipe.fake: pred_spec.copy()}
        if cycle:
            dataset_names[datapipe.cycled] = 'volumes/'+self.model_name+'_enCYCLED'
            array_specs[datapipe.cycled] = pred_spec.copy()

        scan_request = gp.BatchRequest()
        for array in arrays:            
            extents = self.get_extents(side_length, array_name=array.identifier)
            scan_request.add(array, self.common_voxel_size * extents, self.common_voxel_size)

        #Declare new array to write to
        if not hasattr(self, 'compressor'):
            self.compressor = {'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': 64
                }
        
        source_ds = daisy.open_ds(datapipe.src_path, datapipe.real_name)
        datapipe.total_roi = source_ds.data_roi
        for key, name in dataset_names.items():
            write_size = scan_request[key].roi.get_shape()
            daisy.prepare_ds(
                datapipe.out_path,
                name,
                datapipe.total_roi,
                daisy.Coordinate(self.common_voxel_size),
                np.uint8,
                write_size=write_size,
                num_channels=1,
                compressor=self.compressor)

        render_pipe = datapipe.source 
        if datapipe.resample: render_pipe += datapipe.resample
        render_pipe += datapipe.normalize_real
        render_pipe += datapipe.scaleimg2tanh_real

        if datapipe.unsqueeze: # add "channel" dimensions if neccessary, else use z dimension as channel
            render_pipe += datapipe.unsqueeze
        render_pipe += gp.Unsqueeze([datapipe.real]) # add batch dimension

        render_pipe += gp.torch.Predict(self.model,
                                inputs = input_dict,
                                outputs = output_dict,
                                checkpoint = self.checkpoint,
                                array_specs = array_specs, 
                                spawn_subprocess=self.spawn_subprocess
                                )                
        
        if cycle:
            predict_pipe += datapipe.postnet_pipe.cycle
        else:
            predict_pipe += datapipe.postnet_pipe.nocycle
        render_pipe += gp.Squeeze(arrays[1:], axis=0) # remove batch dimension
        
        render_pipe += gp.AsType(datapipe.fake, np.uint8)        
        if cycle:
            render_pipe += gp.AsType(datapipe.cycled, np.uint8)        

        render_pipe += gp.ZarrWrite(
                        dataset_names = dataset_names,
                        output_filename = datapipe.out_path,
                        compression_type = self.compressor
                        # dataset_dtypes = {fake: pred_spec.dtype}
                        )        
        
        render_pipe += gp.Scan(scan_request, num_workers=self.num_workers, cache_size=self.cache_size)

        request = gp.BatchRequest()

        print(f'Full rendering pipeline declared for input type {side}. Building...')
        with gp.build(render_pipe):
            print('Starting full volume render...')
            render_pipe.request_batch(request)
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
