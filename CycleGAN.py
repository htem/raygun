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
from tri_utils import NLayerDiscriminator, NLayerDiscriminator3D, GANLoss, init_weights
from CycleGAN_Model import *
from CycleGAN_LossFunctions import *

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
            side_length=32,# in dataset A sized voxels at output layer - actual used ROI for network input will be bigger for valid padding
            gnet_depth=3, # number of layers in unets (i.e. generators)
            dnet_depth=3, # number of layers in Discriminator networks
            g_downsample_factor=2,
            d_downsample_factor=2,
            g_num_fmaps=16,
            d_num_fmaps=16,
            g_fmap_inc_factor=2,
            g_constant_upsample=True,            
            g_kernel_size_down=None,
            g_kernel_size_up=None,
            d_kernel_size=3, 
            num_epochs=10000,
            batch_size=1,
            num_workers=11,
            cache_size=50,
            spawn_subprocess=False,
            g_init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            d_init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            l1_lambda=10, # Default from CycleGAN paper
            identity_lambda=0.5, # Default from CycleGAN paper
            log_every=100,
            save_every=2000,
            tensorboard_path='./tensorboard/',
            verbose=True,
            checkpoint=None, # Used for prediction/rendering, training always starts from latest
            interp_order=None,
            loss_style='cycle', # supports 'cycle' or 'split'
            min_coefvar=None,
            unet_activation='ReLU',
            residual_unet=False,
            residual_blocks=False,
            padding_unet='same',
            gan_mode='lsgan',
            sampling_bottleneck=False,
            adam_betas = [0.9, 0.999]
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
            self.unet_activation = unet_activation
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
        # for key, loss in self.loss.loss_dict.items():
        #     # self.trainer.summary_writer.add_scalar(key.replace('_', '/'), loss, n_iter)
        #     self.trainer.summary_writer.add_scalar(key, loss, self.n_iter)
        # # self.trainer.summary_writer.add_scalars('Loss', self.loss.loss_dict, n_iter)

        # for array in self.arrays:
        #     if len(self.batch[array].data.shape) > 3: # pull out batch dimension if necessary
        #         img = self.batch[array].data[i].squeeze()
        #     else:
        #         img = self.batch[array].data.squeeze()
        #     if len(img.shape) == 3:
        #         mid = img.shape[0] // 2 # for 3D volume
        #         data = img[mid]
        #     else:
        #         data = img
        #     self.trainer.summary_writer.add_image(array.identifier, data, global_step=self.n_iter, dataformats='HW')

        # try:
        #     self.trainer.summary_writer.add_image('netG1_layer1_gradients', self.netG1[0].l_conv[0].conv_pass[0].weight.grad, global_step=self.n_iter, dataformats='HW')
        #     self.trainer.summary_writer.add_image('netG2_layer1_gradients', self.netG2[0].l_conv[0].conv_pass[0].weight.grad, global_step=self.n_iter, dataformats='HW')
        # except:
        #     logger.warning('Unable to write gradients to tensorboard.')
        # self.trainer.summary_writer.flush()

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

    def get_unet_kernels(self):
        kernel_size_down = self.g_kernel_size_down #<- length of list determines number of conv passes in level
        kernel_size_up = self.g_kernel_size_up

        if kernel_size_down is None:
            kernel_size_down = [[(3,)*self.ndims, (3,)*self.ndims]]*self.gnet_depth
        if kernel_size_up is None:
            kernel_size_up = [[(3,)*self.ndims, (3,)*self.ndims]]*(self.gnet_depth - 1)
        
        return kernel_size_down, kernel_size_up

    def get_valid_padding(self):
        # figure out proper ROI padding for context
        
        kernel_size_down, kernel_size_up = self.get_unet_kernels()
        level_pads = []
        #going down
        for level in np.arange(self.gnet_depth - 1):
            level_pads.append(np.sum(np.array(kernel_size_down[level]) - 1, axis=0) * (self.g_downsample_factor ** level))

        #bottom level
        level_pads.append(np.sum(np.array(kernel_size_down[-1]) - 1, axis=0) * (self.g_downsample_factor ** (self.gnet_depth - 1)))

        #coming up
        for level in np.arange(self.gnet_depth - 1)[::-1]:
            level_pads.append(np.sum(np.array(kernel_size_up[level]) - 1, axis=0) * (self.g_downsample_factor ** level))

        return gp.Coordinate(np.sum(level_pads, axis=0)) // 2 # in voxels per edge
    
    def get_valid_crop_to(self, size=None):
        #returns next valid size that could be cropped to
        pad = self.get_valid_padding()[0] * 2
        if size is None: 
            size = self.side_length - pad
        
        success = self.check_valid_size(size)
        failed = (size - pad) <= 0
        while not success and not failed:
            size -= 1
            success = self.check_valid_size(size) # if successful, final generator output size is returned
            failed = (size - pad) <= 0
        
        if success:
            try: # final test: through discriminator
                shape = (1,1) + (size - pad,) * self.ndims
                temp = torch.rand(*shape)
                _ = self.netD1(temp)
                return size
            except:
                return False
        else:
            return False
            
    def check_valid_size(self, in_size):
        # Checks if a size is a valid input to a generator, and returns the generators output side length if successful
        def _check_size(size):
            return (int(size) == size) and (size > 0)

        size = in_size
        kernel_size_down, kernel_size_up = self.get_unet_kernels()
        down_fac = self.g_downsample_factor
        try:
            #going down
            for level in np.arange(self.gnet_depth - 1):
                size -= np.sum(np.array(kernel_size_down[level]) - 1, axis=0)[0]
                size /= down_fac
                logger.debug(f'Going down level {level} and size {size}')
                assert _check_size(size)

            #bottom level
            size -= np.sum(np.array(kernel_size_down[-1]) - 1, axis=0)[0]
            logger.debug(f'At bottom with size {size}')
            assert _check_size(size)

            #coming up
            for level in np.arange(self.gnet_depth - 1)[::-1]:
                size *= down_fac
                size -= np.sum(np.array(kernel_size_up[level]) - 1, axis=0)[0]
                logger.debug(f'Going up level {level} and size {size}')
                assert _check_size(size)
            
            #final check
            shape = (1,1) + (in_size,) * self.ndims
            _ = self.netG1(torch.rand(*shape))
            return size
        except:
            return False        

    def find_min_valid_size(self, set=True, start_length=None):
        Dnet = self.get_discriminator()
        pad = self.get_valid_padding()[0] * 2        

        success = False
        if start_length is None:
            side_length = self.side_length
        else:
            side_length = start_length
        print('Finding minimum valid input size. This will run until it finds a solution or breaks your computer. Good luck.')
        while not success:
            try:
                out_size = side_length - pad
                assert out_size > 0
                print(f'Side length {side_length} successful on first pass, with result side length {out_size}.')
                out_size = self.get_valid_crop_to(out_size)
                assert out_size
                print(f'Side length {side_length} successful on both passes, with final side length {out_size}.')
                shape = (1,1) + (out_size,) * self.ndims                
                final_size = Dnet(torch.rand(*shape)).shape
                print(f'Side length {side_length} successful on both passes and through discriminator, with final evaluated side length {final_size[-1]}.')
                if set:
                    self.side_length = side_length
                return side_length
            except Exception as e:
                print(e)
                print(f'Side length {side_length} failed.')
                side_length += 1

    def get_generator(self, conf=None): 
        if conf is None: conf = self
        if conf.residual_unet:
            unet = ResidualUNet(
                    in_channels=1,
                    num_fmaps=conf.g_num_fmaps,
                    fmap_inc_factor=conf.g_fmap_inc_factor,
                    downsample_factors=[(conf.g_downsample_factor,)*conf.ndims,] * (conf.gnet_depth - 1),
                    padding=conf.padding_unet,
                    constant_upsample=conf.g_constant_upsample,
                    voxel_size=conf.common_voxel_size[-conf.ndims:],
                    kernel_size_down=conf.g_kernel_size_down,
                    kernel_size_up=conf.g_kernel_size_up,
                    residual=conf.residual_blocks,
                    activation=conf.unet_activation
                    )
            generator = torch.nn.Sequential(
                                unet, 
                                torch.nn.Sigmoid())
        else:
            unet = UNet(
                    in_channels=1,
                    num_fmaps=conf.g_num_fmaps,
                    fmap_inc_factor=conf.g_fmap_inc_factor,
                    downsample_factors=[(conf.g_downsample_factor,)*conf.ndims,] * (conf.gnet_depth - 1),
                    padding=conf.padding_unet,
                    constant_upsample=conf.g_constant_upsample,
                    voxel_size=conf.common_voxel_size[-conf.ndims:],
                    kernel_size_down=conf.g_kernel_size_down,
                    kernel_size_up=conf.g_kernel_size_up,
                    residual=conf.residual_blocks,
                    activation=conf.unet_activation
                    )
            generator = torch.nn.Sequential(
                                unet,
                                ConvPass(conf.g_num_fmaps, 1, [(1,)*conf.ndims], activation=None, padding=conf.padding_unet), 
                                torch.nn.Sigmoid())
        if conf.unet_activation is not None:
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

        self.model = CycleGAN_Model(self.netG1, self.netD1, self.netG2, self.netD2, scale_factor_A, scale_factor_B)

        if self.crop_roi: # Get padding for cropping loss inputs to valid size
            padding = self.get_valid_padding()
        elif self.padding_unet.lower() == 'valid':
            padding = 'valid'
        else:
            padding = None
        # self.l1_loss = torch.nn.SmoothL1Loss() 
        self.l1_loss = torch.nn.L1Loss() 
        self.gan_loss = GANLoss(gan_mode=self.gan_mode)
        if self.loss_style.lower()=='cycle':
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()), lr=self.g_init_learning_rate, betas=self.adam_betas)#TODO: add betas to config variables
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), lr=self.d_init_learning_rate, betas=self.adam_betas)
            self.optimizer = CycleGAN_Optimizer(self.optimizer_G, self.optimizer_D)
            
            self.loss = CycleGAN_Loss(self.l1_loss, self.gan_loss, self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_D, self.optimizer_G, self.ndims, self.l1_lambda, self.identity_lambda, padding, self.gan_mode)
        
        elif self.loss_style.lower()=='split':
        
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)#TODO: add betas to config variables
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=self.g_init_learning_rate, betas=self.adam_betas)#TODO: add betas to config variables
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=self.d_init_learning_rate, betas=self.adam_betas)
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=self.d_init_learning_rate, betas=self.adam_betas)
            self.optimizer = Split_CycleGAN_Optimizer(self.optimizer_G1, self.optimizer_D1, self.optimizer_G2, self.optimizer_D2)

            self.loss = SplitGAN_Loss(self.l1_loss, self.gan_loss, self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_D1, self.optimizer_G1, self.optimizer_D2, self.optimizer_G2, self.ndims, self.l1_lambda, self.identity_lambda, padding, self.gan_mode)

        else:

            print("Unexpected Loss Style. Accepted options are 'cycle' or 'split'")
            raise

    def build_machine(self):       
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
        self.datapipe_A = self.build_datapipe('A')
        self.datapipe_B = self.build_datapipe('B') #{train_pipe: train_pipe, src_pipe: src_pipe, reject: reject, resample: resample, augment: augment, unsqueeze: unsqueeze}

    def build_datapipe(self, side):
        side = side.upper() # ensure uppercase
        src_voxel_size = getattr(self, side+'_voxel_size')
        
        # declare arrays to use in the pipelines
        array_names = ['real_', 
                        'fake_', 
                        'cycled_']
        if getattr(self, f'mask_{side}_name') is not None: 
            array_names += ['mask_']
            masked = True

        for array in array_names:
            array_name = array + side
            array_key = gp.ArrayKey(array_name.upper())
            locals()[array[:-1]] = array_key # set local variable keys
            setattr(self, array_name, array_key) # add ArrayKeys to object
            if 'mask' not in array:            
                setattr(self, 'normalize_'+array_name, gp.Normalize(array_key))#add normalizations, if appropriate        
        
        #Setup sources and resampling nodes
        if self.common_voxel_size != src_voxel_size:
            real_src = gp.ArrayKey(f'REAL_{side}_SRC')
            setattr(self, f'real_{side}_src', real_src)
            resample = gp.Resample(real_src, self.common_voxel_size, real, ndim=self.ndims, interp_order=self.interp_order)
            if masked: 
                mask_src = gp.ArrayKey(f'MASK_{side}_SRC')
                setattr(self, f'mask_{side}_src', mask_src)
                resample += gp.Resample(mask_src, self.common_voxel_size, mask, ndim=self.ndims, interp_order=self.interp_order)
        else:            
            real_src = real
            resample = None
            if masked: 
                mask_src = mask

        # setup data sources
        src = getattr(self, 'src_'+side)# the zarr container
        src_name = getattr(self, side+'_name')
        if masked: 
            mask_name = getattr(self, f'mask_{side}_name')
            source = gp.ZarrSource(    # add the data source
                    src,  
                {   real_src: src_name,
                    mask_src: mask_name,
                    },  # which dataset to associate to the array key
                {   real_src: gp.ArraySpec(interpolatable=True, voxel_size=src_voxel_size),
                    mask_src: gp.ArraySpec(interpolatable=False), 
                    }  # meta-information
            )
        else:                        
            source = gp.ZarrSource(    # add the data source
                    src,  
                {   real_src: src_name,
                    },  # which dataset to associate to the array key
                {   real_src: gp.ArraySpec(interpolatable=True, voxel_size=src_voxel_size),
                    }  # meta-information
            )

        #make initial pipe section for A: TODO: Make min_masked part of config
        src_pipe = source

        if masked:
            reject = gp.Reject(mask=self.mask_A_src, min_masked=0.999)

        if self.min_coefvar:
            if not hasattr(locals(), 'reject'):
                reject = gp.RejectConstant(real_src, min_coefvar=self.min_coefvar)
            else:
                reject += gp.RejectConstant(real_src, min_coefvar=self.min_coefvar)
        
        if not hasattr(locals(), 'reject'):
            reject = None

        augment = gp.SimpleAugment(mirror_only=self.augment_axes, transpose_only=self.augment_axes)
        augment += getattr(self, 'normalize_real_'+side)    
        augment += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
            control_point_spacing=self.side_length//2,
            # jitter_sigma=(5.0,)*self.ndims,
            jitter_sigma=(0., 5.0, 5.0,)[-self.ndims:],
            rotation_interval=(0, math.pi/2),
            subsample=4,
            spatial_dims=self.ndims
            )

        # add "channel" dimensions if neccessary, else use z dimension as channel
        if self.ndims == len(self.common_voxel_size):
            unsqueeze = gp.Unsqueeze([real])
        else:
            unsqueeze = None

        # Make training datapipe
        train_pipe = src_pipe + gp.RandomLocation()
        if reject:
            train_pipe += reject
        if resample:
            train_pipe += resample
        train_pipe += augment
        if unsqueeze:
            train_pipe += unsqueeze # add "channel" dimensions if neccessary, else use z dimension as channel
        train_pipe += gp.Stack(self.batch_size)# add "batch" dimensions    
        setattr(self, 'pipe_'+side, train_pipe)

        return {train_pipe: train_pipe, src_pipe: src_pipe, reject: reject, resample: resample, augment: augment, unsqueeze: unsqueeze}

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
        # remove "channel" dimensions if neccessary
        if self.ndims == len(self.common_voxel_size):
            self.training_pipeline += gp.Squeeze([self.real_A, 
                                            self.fake_A, 
                                            self.cycled_A, 
                                            self.real_B, 
                                            self.fake_B, 
                                            self.cycled_B
                                            ], axis=1) # remove channel dimension for grayscale
        if self.batch_size == 1:
            self.training_pipeline += gp.Squeeze([self.real_A, 
                                            self.fake_A, 
                                            self.cycled_A, 
                                            self.real_B, 
                                            self.fake_B, 
                                            self.cycled_B
                                            ], axis=0)
        self.training_pipeline += self.normalize_fake_B + self.normalize_cycled_A
        self.training_pipeline += self.normalize_fake_A + self.normalize_cycled_B
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
                if hasattr(self.loss, 'loss_dict'):
                    print(self.loss.loss_dict)
                if i % self.log_every == 0:
                    self.batch_tBoard_write()
        return self.batch
        
    def test_prediction(self, side='A', side_length=None):
        #set model into evaluation mode
        self.model.eval()
        # model_outputs = {
        #     0: self.fake_B,
        #     1: self.cycled_B,
        #     2: self.fake_A,
        #     3: self.cycled_A}

        #{train_pipe: train_pipe, src_pipe: src_pipe, reject: reject, resample: resample, augment: augment, unsqueeze: unsqueeze}
        datapipe = getattr(self, 'datapipe_'+side)
        for key, value in datapipe:
            locals()[key] = value
        
        for array in ['real_', 'fake_', 'cycle_']:
            locals()[array[:-1]] = getattr(self, array+side)
            locals()['normalize_'+array[:-1]] = getattr(self, 'normalize_'+array+side)

        input_dict = {'real_'+side: real}

        if in_type=='A':
            output_dict = { 0: fake,
                            3: cycled
            }
        else:           
            output_dict = { 2: fake,
                            1: cycled
            }

        predict_pipe = src_pipe + gp.RandomLocation() + reject + resample + normalize_real

        if unsqueeze: # add "channel" dimensions if neccessary, else use z dimension as channel
            predict_pipe += unsqueeze
        predict_pipe += gp.Unsqueeze([real]) # add batch dimension

        predict_pipe += gp.torch.Predict(self.model,
                                inputs = input_dict,
                                outputs = output_dict,
                                checkpoint = self.checkpoint
                                )
        
        predict_pipe += gp.Squeeze([real, fake, cycled], axis=1) # remove "channel" dimension
        predict_pipe += gp.Squeeze([real, fake, cycled], axis=0) # remove batch dimension
        predict_pipe += normalize_fake + normalize_cycled

        request = gp.BatchRequest()
        for array in [real, fake, cycled]:            
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
        
        #{train_pipe: train_pipe, src_pipe: src_pipe, reject: reject, resample: resample, augment: augment, unsqueeze: unsqueeze}
        datapipe = getattr(self, 'datapipe_'+side)
        for key, value in datapipe:
            locals()[key] = value
        src_path = getattr(self, 'src_'+side)
        out_path = getattr(self, side+'_out_path')
        real_name = getattr(self, side+'_name')
        
        for array in ['real_', 'fake_', 'cycle_']:
            locals()[array[:-1]] = getattr(self, array+side)
            locals()['normalize_'+array[:-1]] = getattr(self, 'normalize_'+array+side)

        input_dict = {'real_'+side: real}

        if in_type=='A':
            output_dict = { 0: fake,
                            3: cycled
            }
        else:           
            output_dict = { 2: fake,
                            1: cycled
            }

        # set prediction spec 
        if src_pipe.spec is None:
            data_file = zarr.open(src_path)
            pred_spec = src_pipe._Hdf5LikeSource__read_spec(real, data_file, real_name).copy()
        else:
            pred_spec = src_pipe.spec[real].copy()        
        pred_spec.voxel_size = self.common_voxel_size
        pred_spec.dtype = normalize_fake.dtype

        arrays = [real, fake]
        dataset_names = {fake: 'volumes/'+self.model_name+'_enFAKE'}
        array_specs = {fake: pred_spec.copy()}
        if cycle:
            arrays += [cycled]
            dataset_names[cycled] = 'volumes/'+self.model_name+'_enCYCLED'
            array_specs[cycled] = pred_spec.copy()

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
        
        source_ds = daisy.open_ds(src_path, real_name)
        total_roi = source_ds.data_roi
        for key, name in dataset_names.items():
            write_size = scan_request[key].roi.get_shape()
            daisy.prepare_ds(
                out_path,
                name,
                total_roi,
                daisy.Coordinate(self.common_voxel_size),
                np.uint8,
                write_size=write_size,
                num_channels=1,
                compressor=self.compressor)

        render_pipe = src_pipe + resample + normalize_real

        if unsqueeze: # add "channel" dimensions if neccessary, else use z dimension as channel
            render_pipe += unsqueeze
        render_pipe += gp.Unsqueeze([real]) # add batch dimension

        render_pipe += gp.torch.Predict(self.model,
                                inputs = input_dict,
                                outputs = output_dict,
                                checkpoint = self.checkpoint,
                                array_specs = array_specs, 
                                spawn_subprocess=self.spawn_subprocess
                                )
        
        render_pipe += gp.Squeeze(arrays[1:], axis=1) # remove "channel" dimension
        render_pipe += gp.Squeeze(arrays[1:], axis=0) # remove batch dimension
        
        render_pipe += normalize_fake + gp.AsType(fake, np.uint8)        
        if cycle:
            render_pipe += normalize_cycled + gp.AsType(cycled, np.uint8)        

        render_pipe += gp.ZarrWrite(
                        dataset_names = dataset_names,
                        output_filename = out_path,
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


class CycleGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G, optimizer_D):
        super(CycleGAN_Optimizer, self).__init__()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass

class Split_CycleGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G1, optimizer_D1, optimizer_G2, optimizer_D2):
        super(Split_CycleGAN_Optimizer, self).__init__()
        self.optimizer_G1 = optimizer_G1
        self.optimizer_D1 = optimizer_D1
        self.optimizer_G2 = optimizer_G2
        self.optimizer_D2 = optimizer_D2

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass