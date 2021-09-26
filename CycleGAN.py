# !conda activate n2v
import numpy as np
from matplotlib import pyplot as plt
# from numpy.lib.utils import source
import torch
import glob
import re
import zarr

from funlib.learn.torch.models import UNet, ConvPass
import gunpowder as gp
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN', 'INFO')

import math
import sys
import functools
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

from tri_utils import NLayerDiscriminator3D, GANLoss, init_weights, UnetGenerator3D

class CycleGAN(): #TODO: Just pass config file or dictionary
    def __init__(self,
            src_A, #EXPECTS ZARR VOLUME
            src_B,
            voxel_size, # voxel size of src_B (for each dimension)
            AB_voxel_ratio=1, #determines whether to add up/downsampling node (always rendered to B's voxel_size), (int or tuple of ints)
            A_name='raw',
            B_name='raw',
            mask_A_name='mask', # expects mask to be in same place as real zarr
            mask_B_name='mask',
            A_out_path=None,
            B_out_path=None,
            model_name='CycleGun',
            model_path='./models/',
            side_length=64,#12 # in voxels for prediction (i.e. network output) - actual used ROI for network input will be bigger for valid padding
            gnet_depth=4, # number of layers in unets (i.e. generators)
            dnet_depth=3, # number of layers in Discriminator networks
            g_downsample_factor=2,
            d_downsample_factor=2,
            g_conv_padding='valid',
            d_conv_padding='valid',
            g_num_fmaps=32,
            d_num_fmaps=32,
            g_fmap_inc_factor=2,
            d_fmap_inc_factor=2,
            g_constant_upsample=True,
            d_constant_upsample=True,
            g_conv_passes=2, 
            d_conv_passes=2,
            g_kernel_size=3,
            d_kernel_size=3, 
            num_epochs=10000,
            batch_size=1,
            num_workers=11,
            g_init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            d_init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            log_every=100,
            save_every=2000,
            tensorboard_path='./tensorboard/',
            verbose=True,
            checkpoint=None # Used for prediction/rendering, training always starts from latest
            ):
            self.src_A = src_A
            self.src_B = src_B
            self.voxel_size = gp.Coordinate(voxel_size)
            self.AB_voxel_ratio = AB_voxel_ratio
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
            self.g_conv_padding = g_conv_padding
            self.d_conv_padding = d_conv_padding
            self.g_num_fmaps = g_num_fmaps
            self.d_num_fmaps = d_num_fmaps
            self.g_fmap_inc_factor = g_fmap_inc_factor
            self.d_fmap_inc_factor = d_fmap_inc_factor
            self.g_constant_upsample = g_constant_upsample
            self.d_constant_upsample = d_constant_upsample
            self.g_conv_passes = g_conv_passes 
            self.d_conv_passes = d_conv_passes 
            self.g_kernel_size = g_kernel_size 
            self.d_kernel_size = d_kernel_size 
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.g_init_learning_rate = g_init_learning_rate
            self.d_init_learning_rate = d_init_learning_rate
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
            self.build_pipeline_parts()
            self.training_pipeline = None

    def set_device(self, id=0):
        torch.cuda.set_device(id)   
    
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
            self.col_dict = {'REAL':0, 'FAKE':1, 'CYCL':2, 'MASK':3}
        fig, axes = plt.subplots(2, len(self.col_dict), figsize=(30*2, 30*len(self.col_dict)))
        for array, value in batch.items():
            if ('cropped'.upper() in array.identifier) or ('real'.upper() not in array.identifier):
                label = array.identifier
                c = self.col_dict[label[:4]]
                r = (int('_B' in label) + int('FAKE' in label)) % 2
                img = value.data[i].squeeze()
                mid = img.shape[0] // 2 # TODO: assumes 3D volume
                data = img[mid]
                axes[r, c].imshow(data, cmap='gray', vmin=0, vmax=1)
                axes[r, c].set_title(label)
    
    def get_validation_loss(self):
        validation_loss = self.loss.validation(
                        self.batch[self.real_A_cropped].data, 
                        self.batch[self.fake_A].data, 
                        self.batch[self.cycled_A].data, 
                        self.batch[self.real_B_cropped].data, 
                        self.batch[self.fake_B].data, 
                        self.batch[self.cycled_B].data, 
                        not self.batch[self.mask_A].data, 
                        not self.batch[self.mask_B].data)
        self.validation_loss = validation_loss
        return validation_loss

    def batch_tBoard_write(self, i=0):
        for array in self.arrays:
            if ('cropped'.upper() in array.identifier) or ('real'.upper() not in array.identifier):
                img = self.batch[array].data[i].squeeze()
                mid = img.shape[0] // 2 # TODO: assumes 3D volume
                self.trainer.summary_writer.add_image(array.identifier, img[mid], global_step=self.trainer.iteration, dataformats='HW')
        # TODO:
        # validation_loss = self.get_validation_loss()
        # self.trainer.summary_writer.add_scalar('validation_loss', validation_loss, self.trainer.iteration)

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
    
    def build_pipeline_parts(self):        
        # declare arrays to use in the pipelines
        self.array_names = ['real_A', 'real_A_cropped', 'mask_A', 'fake_A', 'cycled_A', 'real_B', 'real_B_cropped', 'mask_B', 'fake_B', 'cycled_B']#, 'gradients_G1', 'gradients_G2']
        # [from source A, cropped version to match netG2 output size, mask of source for training (cropped size), netG2 output, netG2(netG1(real_A)) output,...for other side...] #TODO: add gradients for network training debugging
        self.arrays = []
        for array in self.array_names:
            setattr(self, array, gp.ArrayKey(array.upper()))
            self.arrays.append(getattr(self, array))
            #add normalizations, if appropriate
            if 'mask' not in array:
                setattr(self, 'normalize_'+array, gp.Normalize(getattr(self, array)))
        
        # automatically generate some config variables
        self.gen_context_side_length()
        # self.crops = {self.raw:self.prediction}
        
        if not self.AB_voxel_ratio == 1:
            self.real_A_src = gp.ArrayKey('REAL_A_SRC')
            self.real_A_cropped_src = gp.ArrayKey('REAL_A_CROPPED_SRC')
            self.mask_A_src = gp.ArrayKey('MASK_A_SRC')
            if self.AB_voxel_ratio < 1:
                self.resample = gp.DownSample(self.real_A_src, self.AB_voxel_ratio, self.real_A)
                self.resample += gp.DownSample(self.real_A_cropped_src, self.AB_voxel_ratio, self.real_A_cropped)
                self.resample += gp.DownSample(self.mask_A_src, self.AB_voxel_ratio, self.mask_A)
            if self.AB_voxel_ratio > 1:
                self.resample = gp.UpSample(self.real_A_src, self.AB_voxel_ratio, self.real_A)
                self.resample += gp.UpSample(self.real_A_cropped_src, self.AB_voxel_ratio, self.real_A_cropped)
                self.resample += gp.UpSample(self.mask_A_src, self.AB_voxel_ratio, self.mask_A)
        else:            
            self.real_A_src = self.real_A
            self.real_A_cropped_src = self.real_A_cropped
            self.mask_A_src = self.mask_A
            self.resample = None

        # setup data sources
        self.source_A = gp.ZarrSource(    # add the data source
            self.src_A,  # the zarr container
            {   self.real_A_src: self.A_name,
                self.real_A_cropped_src: self.A_name,
                self.mask_A_src: self.mask_A_name,
                },  # which dataset to associate to the array key
            {   self.real_A_src: gp.ArraySpec(interpolatable=True),
                self.real_A_cropped_src: gp.ArraySpec(interpolatable=True),
                self.mask_A_src: gp.ArraySpec(interpolatable=False), #TODO: Determine whether to include voxel_size
                }  # meta-information
        )
        self.source_A += self.resample

        self.source_B = gp.ZarrSource(    # add the data source
            self.src_B,  # the zarr container
            {   self.real_B: self.B_name,
                self.real_B_cropped: self.B_name,
                self.mask_B: self.mask_B_name,
                },  # which dataset to associate to the array key
            {   self.real_B: gp.ArraySpec(interpolatable=True),
                self.real_B_cropped: gp.ArraySpec(interpolatable=True),
                self.mask_B: gp.ArraySpec(interpolatable=False), #TODO: Determine whether to include voxel_size
                }  # meta-information
        )

        # get performance stats
        self.performance = gp.PrintProfilingStats(every=self.log_every)

        # stack for batches
        # self.stack = gp.Stack(self.batch_size) # TODO: Determine if removing increases speed

        # setup a cache
        self.cache = gp.PreCache(num_workers=self.num_workers)#os.cpu_count())

        # define our network model for training
        self.setup_networks()

    def setup_networks(self):
        #For netG1:
        unet = UNet(
                in_channels=1,
                num_fmaps=self.g_num_fmaps,
                fmap_inc_factor=self.g_fmap_inc_factor,
                downsample_factors=[(self.g_downsample_factor,)*3,] * (self.gnet_depth - 1), #TODO: make work for arbitrary dimensionality
                padding=self.g_conv_padding,
                constant_upsample=self.g_constant_upsample,
                voxel_size=self.voxel_size # set for each dataset
                )
        self.netG1 = torch.nn.Sequential(
                            unet,
                            ConvPass(self.g_num_fmaps, 1, [(1,)*3], activation=None),
                            torch.nn.Sigmoid())
        init_weights(self.netG1, init_type='normal', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE

        #For netG2:
        unet = UNet(
                in_channels=1,
                num_fmaps=self.g_num_fmaps,
                fmap_inc_factor=self.g_fmap_inc_factor,
                downsample_factors=[(self.g_downsample_factor,)*3,] * (self.gnet_depth - 1), #TODO: make work for arbitrary dimensionality
                padding=self.g_conv_padding,
                constant_upsample=self.g_constant_upsample,
                voxel_size=self.voxel_size # set for each dataset
                )        
        self.netG2 = torch.nn.Sequential(
                            unet,
                            ConvPass(self.g_num_fmaps, 1, [(1,)*3], activation=None),
                            torch.nn.Sigmoid())
        init_weights(self.netG2, init_type='normal', init_gain=0.05) #TODO: MAY WANT TO ADD TO CONFIG FILE

        #For netD1:
        norm_layer = functools.partial(torch.nn.InstanceNorm3d, affine=False, track_running_stats=False)
        self.netD1 = NLayerDiscriminator3D(input_nc=1, 
                                        ndf=self.d_num_fmaps, 
                                        n_layers=self.dnet_depth, 
                                        norm_layer=norm_layer,
                                        downsampling_kw=self.d_downsample_factor, 
                                        kw=self.d_kernel_size,
                                 )
        init_weights(self.netD1, init_type='normal')

        #For netD2:
        norm_layer = functools.partial(torch.nn.InstanceNorm3d, affine=False, track_running_stats=False)
        self.netD2 = NLayerDiscriminator3D(input_nc=1, 
                                        ndf=self.d_num_fmaps, 
                                        n_layers=self.dnet_depth, 
                                        norm_layer=norm_layer,
                                        downsampling_kw=self.d_downsample_factor, 
                                        kw=self.d_kernel_size,
                                 )
        init_weights(self.netD2, init_type='normal')

    def setup_model(self):
        if not hasattr(self, 'netG1'):
            self.setup_networks()
        self.model = CycleGAN_Model(self.netG1, self.netD1, self.netG2, self.netD2)

        self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=self.g_init_learning_rate, betas=(0.95, 0.999))#TODO: add betas to config variables
        self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=self.d_init_learning_rate, betas=(0.95, 0.999))
        self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=self.g_init_learning_rate, betas=(0.95, 0.999))#TODO: add betas to config variables
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=self.d_init_learning_rate, betas=(0.95, 0.999))
        self.optimizer = CycleGAN_Optimizer(self.optimizer_G1, self.optimizer_D1, self.optimizer_G2, self.optimizer_D2)

        self.l1_loss = torch.nn.L1Loss()
        self.gan_loss = GANLoss(gan_mode='lsgan')
        self.loss = CycleGAN_Loss(self.l1_loss, self.gan_loss, self.netD1, self.netG1, self.netD2, self.netG2, self.optimizer_D1, self.optimizer_G1, self.optimizer_D2, self.optimizer_G2) #TODO: add l1_lambda=### to config

    def build_training_pipeline(self):
        # # add augmentations
        # self.simple_augment = gp.SimpleAugment()
        # self.elastic_augment = gp.ElasticAugment(
        #     # control_point_spacing=(64, 64),
        #     # control_point_spacing=(48*30, 48*30, 48*30),
        #     control_point_spacing=(48, 48, 48),
        #     jitter_sigma=(5.0, 5.0, 5.0),
        #     rotation_interval=(0, math.pi/2),
        #     subsample=4,
        #     )
        # # pipeline += gp.IntensityAugment(
        # #     source,
        # #     scale_min=0.8,
        # #     scale_max=1.2,
        # #     shift_min=-0.2,
        # #     shift_max=0.2)

        # # add a RandomLocation node to the pipeline to randomly select a sample
        # self.random_location = gp.RandomLocation()

        self.setup_model()

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
                                # 'fake_B': self.fake_B,
                                # 'cycled_B': self.cycled_B,
                                # 'fake_A': self.fake_A,
                                # 'cycled_A': self.cycled_A
                                0: self.fake_B,
                                1: self.cycled_B,
                                2: self.fake_A,
                                3: self.cycled_A
                            },
                            loss_inputs = {
                                'real_A': self.real_A_cropped,
                                'fake_A': self.fake_A,
                                'cycled_A': self.cycled_A,
                                'real_B': self.real_B_cropped,
                                'fake_B': self.fake_B,
                                'cycled_B': self.cycled_B,
                                'mask_A': self.mask_A,
                                'mask_B': self.mask_B
                            },
                            log_dir=self.tensorboard_path,
                            log_every=self.log_every,
                            checkpoint_basename=self.model_path+self.model_name,
                            save_every=self.save_every
                            )

        # create request
        self.train_request = gp.BatchRequest()
        for array in self.arrays:
            if ('cropped'.upper() in array.identifier) or ('real'.upper() not in array.identifier):
                self.train_request.add(array, self.voxel_size*self.side_length)
            else:
                self.train_request.add(array, self.voxel_size*self.context_side_length)

        #make initial pipe section for A: TODO: Make min_masked part of config
        pipe_A = self.source_A + gp.RandomLocation(min_masked=0.5, mask=self.mask_A) + self.normalize_real_A + self.normalize_real_A_cropped + gp.SimpleAugment()
        pipe_A += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
            # control_point_spacing=(64, 64),
            # control_point_spacing=(48*30, 48*30, 48*30),
            control_point_spacing=(48, 48, 48),
            jitter_sigma=(5.0, 5.0, 5.0),
            rotation_interval=(0, math.pi/2),
            subsample=4,
            )
        # add "channel" dimensions
        pipe_A += gp.Unsqueeze([self.real_A, self.real_A_cropped]) #MAY NEED TO ADD MASK TO LIST
        # add "batch" dimensions
        pipe_A += gp.Unsqueeze([self.real_A, self.real_A_cropped]) #MAY NEED TO ADD MASK TO LIST

        #make initial pipe section for B: TODO: Make min_masked part of config
        pipe_B = self.source_B + gp.RandomLocation(min_masked=0.5, mask=self.mask_B) + self.normalize_real_B + self.normalize_real_B_cropped + gp.SimpleAugment()
        pipe_B += gp.ElasticAugment( #TODO: MAKE THESE SPECS PART OF CONFIG
            # control_point_spacing=(64, 64),
            # control_point_spacing=(48*30, 48*30, 48*30),
            control_point_spacing=(48, 48, 48),
            jitter_sigma=(5.0, 5.0, 5.0),
            rotation_interval=(0, math.pi/2),
            subsample=4,
            )
        # add "channel" dimensions
        pipe_B += gp.Unsqueeze([self.real_B, self.real_B_cropped]) #MAY NEED TO ADD MASK TO LIST
        # add "batch" dimensions
        pipe_B += gp.Unsqueeze([self.real_B, self.real_B_cropped]) #MAY NEED TO ADD MASK TO LIST

        # assemble pipeline
        self.training_pipeline = (pipe_A + pipe_B) #+ gp.MergeProvider #merge upstream pipelines for two sources
        self.training_pipeline += self.cache + self.trainer
        self.training_pipeline += gp.Squeeze([self.real_A, self.real_A_cropped, self.fake_A, self.cycled_A, self.real_B, self.real_B_cropped, self.fake_B, self.cycled_B], axis=0)
        self.training_pipeline += gp.Squeeze([self.real_A, self.real_A_cropped, self.fake_A, self.cycled_A, self.real_B, self.real_B_cropped, self.fake_B, self.cycled_B], axis=0)
        self.training_pipeline += self.performance
    
    def gen_context_side_length(self):
        # figure out proper ROI padding for context for the UNet generators
        if self.g_conv_padding == 'valid':
            self.context_side_length = 2 * np.sum([(self.g_conv_passes * (self.g_kernel_size - 1)) * (2 ** level) for level in np.arange(self.gnet_depth - 1)]) + (self.g_conv_passes * (self.g_kernel_size - 1)) * (2 ** (self.gnet_depth - 1)) + (self.g_conv_passes * (self.g_kernel_size - 1)) + self.side_length
        else:
            self.context_side_length = self.side_length

    def test_train(self):
        if self.training_pipeline is None:
            self.build_training_pipeline()
        self.model.train()
        with gp.build(self.training_pipeline):
            self.batch = self.training_pipeline.request_batch(self.train_request)
        self.batch_show()
        return self.batch

    def train(self):
        if self.training_pipeline is None:
            self.build_training_pipeline()
        self.model.train()
        with gp.build(self.training_pipeline):
            for i in tqdm(range(self.num_epochs)):
                self.batch = self.training_pipeline.request_batch(self.train_request)
                if i % self.log_every == 0:
                    self.batch_tBoard_write()
        return self.batch
        
    def test_prediction(self, n=1):
        #set model into evaluation mode
        self.model.eval()

        unsqueeze = gp.Unsqueeze([self.raw])
        stack = gp.Stack(n)

        self.predict = gp.torch.Predict(self.model,
                                inputs = {'input': self.raw},
                                outputs = {0: self.prediction},
                                checkpoint = self.checkpoint
                                )

        request = gp.BatchRequest()
        request.add(self.prediction, self.voxel_size*self.side_length)
        request.add(self.gt, self.voxel_size*self.side_length)
        request.add(self.raw, self.voxel_size*self.context_side_length)

        predicter = (self.source + 
                    self.normalize_raw + 
                    self.normalize_gt +
                    self.random_location + 
                    unsqueeze +
                    stack +
                    self.predict +
                    self.normalize_pred)

        with gp.build(predicter):
            self.batch = predicter.request_batch(request)

        self.batch_show()
        return self.batch

    def render_full(self):
        #set model into evaluation mode
        self.model.eval()

        unsqueeze = gp.Unsqueeze([self.raw])
        squeeze_1 = gp.Squeeze([self.prediction])
        squeeze_2 = gp.Squeeze([self.prediction])

        # set prediction spec
        context = self.voxel_size * (self.context_side_length - self.side_length) // 2
        if self.source.spec is None:
            data_file = zarr.open(self.src)
            pred_spec = self.source._Hdf5LikeSource__read_spec(self.raw, data_file, self.raw_name).copy()
        else:
            pred_spec = self.source.spec[self.raw].copy()        
        pred_spec.roi.grow(-context, -context)
        pred_spec.dtype = self.normalize_pred.dtype

        self.predict = gp.torch.Predict(self.model,
                                inputs = {'input': self.raw},
                                outputs = {0: self.prediction},
                                checkpoint = self.checkpoint,
                                array_specs = {self.prediction: pred_spec}
                                )

        scan_request = gp.BatchRequest()
        scan_request.add(self.prediction, self.voxel_size*self.side_length)
        scan_request.add(self.raw, self.voxel_size*self.context_side_length)
        scan = gp.Scan(scan_request, num_workers=self.num_workers)#os.cpu_count())

        destination = gp.ZarrWrite(
                        dataset_names = {
                            self.prediction: self.model_name
                            },
                        output_filename = self.out_path,
                        dataset_dtypes = {self.prediction: pred_spec.dtype}
                        )

        renderer = (self.source + 
                    self.normalize_raw +  
                    unsqueeze +
                    self.stack +
                    self.predict +
                    squeeze_1 +
                    squeeze_2 +
                    self.normalize_pred +
                    destination +
                    scan +
                    self.performance)

        request = gp.BatchRequest()

        print('Full rendering pipeline declared. Building...')
        with gp.build(renderer):
            print('Starting full volume render...')
            renderer.request_batch(request)
            print('Finished.')


class CycleGAN_Model(torch.nn.Module):
    def __init__(self, netG1, netD1, netG2, netD2):
        super(CycleGAN_Model, self).__init__()
        self.netG1 = netG1
        self.netD1 = netD1
        self.netG2 = netG2
        self.netD2= netD2

    def forward(self, real_A, real_B):
        fake_B = self.netG1(real_A)
        cycled_A = self.netG2(fake_B)
        fake_A = self.netG2(real_B)
        cycled_B = self.netG1(fake_A)

        return fake_B, cycled_B, fake_A, cycled_A


class CycleGAN_Optimizer(torch.nn.Module):
    def __init__(self, optimizer_G1, optimizer_D1, optimizer_G2, optimizer_D2):
        super(CycleGAN_Optimizer, self).__init__()
        self.optimizer_G1 = optimizer_G1
        self.optimizer_D1 = optimizer_D1
        self.optimizer_G2 = optimizer_G2
        self.optimizer_D2 = optimizer_D2

    def step(self):
        """Dummy step pass for Gunpowder's Train node step() call"""
        pass


class CycleGAN_Loss(torch.nn.Module):
    def __init__(self, l1_loss, gan_loss, netD1, netG1, netD2, netG2, optimizer_D1, optimizer_G1, optimizer_D2, optimizer_G2, l1_lambda=100
                 ):
        super(CycleGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = gan_loss
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_D1 = optimizer_D1
        self.optimizer_G1 = optimizer_G1
        self.optimizer_D2 = optimizer_D2
        self.optimizer_G2 = optimizer_G2
        self.l1_lambda = l1_lambda

    def backward_D(self, Dnet, real, fake, cycled):
        # Real
        pred_real = Dnet(real)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = Dnet(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        # Cycled; stop backprop to the generator by detaching cycled
        pred_cycled = Dnet(cycled.detach())
        loss_D_cycled = self.gan_loss(pred_cycled, False)

        loss_D = (loss_D_real + loss_D_fake + loss_D_cycled) / 3
        loss_D.backward()
        return loss_D

    def backward_Ds(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D1.zero_grad()     # set D's gradients to zero
        self.optimizer_D2.zero_grad()     # set D's gradients to zero

        #Do D1 first
        loss_D1 = self.backward_D(self.netD1, real_B, fake_B, cycled_B)
        self.optimizer_D1.step()          # update D's weights

        #Then D2
        loss_D2 = self.backward_D(self.netD2, real_A, fake_A, cycled_A)
        self.optimizer_D2.step()

        #return losses
        return loss_D1, loss_D2

    def backward_G(self, Dnet, fake, cycled, cycle_loss):
        """Calculate GAN and L1 loss for the generator"""
        # Include L1 loss for forward and reverse cycle consistency
        loss_G = cycle_loss.clone() #TODO: verify, maybe should be detach() instead?

        # First, G(A) should fake the discriminator
        pred_fake = Dnet(fake)
        loss_G += self.gan_loss(pred_fake, True)

        # Second, G(F(B)) should also fake the discriminator TODO: VERIFY THIS (maybe unnecessary)
        pred_cycled = Dnet(cycled)
        loss_G += self.gan_loss(pred_cycled, True)

        # calculate gradients
        loss_G.backward()
        return loss_G

    def backward_Gs(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero

        #get cycle loss for both directions (i.e. real == cycled, a.k.a. real_A == netG2(netG1(real_A)) for A and B)
        cycle_loss = self.l1_lambda * (self.l1_loss(real_A, cycled_A) + self.l1_loss(real_B, cycled_B))

        #Then G1 first
        loss_G1 = self.backward_G(self.netD1, fake_B, cycled_B, cycle_loss)                   # calculate gradient for G
        self.optimizer_G1.step()             # udpate G's weights

        #Then G2
        loss_G2 = self.backward_G(self.netD2, fake_A, cycled_A, cycle_loss)                   # calculate gradient for G
        self.optimizer_G2.step()             # udpate G's weights

        #return losses
        return cycle_loss, loss_G1, loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B, mask_A, mask_B):

        real_A_mask = real_A * mask_A
        cycled_A_mask = cycled_A * mask_A
        fake_A_mask = fake_A * mask_B # masked based on mask from "real" version of array before generator pass
        real_B_mask = real_B * mask_B
        cycled_B_mask = cycled_B * mask_B
        fake_B_mask = fake_B * mask_A

        # # update Ds
        loss_D1, loss_D2 = self.backward_Ds(real_A_mask, fake_A_mask, cycled_A_mask, real_B_mask, fake_B_mask, cycled_B_mask)

        # update Gs
        cycle_loss, loss_G1, loss_G2 = self.backward_Gs(real_A_mask, fake_A_mask, cycled_A_mask, real_B_mask, fake_B_mask, cycled_B_mask)

        self.loss_dict = {
            'loss_D1': float(loss_D1),
            'loss_D2': float(loss_D2),
            'loss_cycle': float(cycle_loss),
            'loss_G1': float(loss_G1),
            'loss_G2': float(loss_G2),
        }

        total_loss = loss_D1 + loss_D2 + cycle_loss + loss_G1 + loss_G2
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        print(self.loss_dict)
        return total_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def validation(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B, mask_A, mask_B):
        with torch.no_grad(): #MAY NOT WORK
            validation_loss = self.forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B, mask_A, mask_B)
        return validation_loss

# TODO:
# if __name__ == '__main__':

#     import json    
#     config = json.loads(sys.argv[0])
#     print('Loaded config...')

#     cycleGAN = CycleGAN(config)

#     try:
#         cycleGAN.num_epochs = int(sys.argv[1])
#     except:
#         pass    

#     if 'test' in sys.argv:
#         cycleGAN.num_workers = 1        
#         cycleGAN.test_train()
#     else:
#         cycleGAN.train()
