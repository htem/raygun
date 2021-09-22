# !conda activate n2v
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import glob
import re
import zarr

from funlib.learn.torch.models import UNet, ConvPass
import gunpowder as gp
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN', 'INFO')

class CycleGAN(): #TODO: REWRITE FOR CYCLEGAN (copied from CARE)
    def __init__(self,
            src, #EXPECTS ZARR VOLUME
            voxel_size,
            gt_name='gt',
            raw_name='train',
            out_path=None,
            model_name='CARE',
            model_path='./models/',
            side_length=64,#12 # in voxels for prediction (i.e. network output) - actual used ROI for network input will be bigger for valid padding
            unet_depth=4, # number of layers in unet
            downsample_factor=2,
            conv_padding='valid',
            num_fmaps=12,
            fmap_inc_factor=5,
            constant_upsample=True,
            num_epochs=10000,
            batch_size=1,
            init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            log_every=100,
            save_every=2000,
            tensorboard_path='./tensorboard/',
            verbose=True,
            checkpoint=None # Used for prediction/rendering, training always starts from latest
            ):
            self.src = src
            self.voxel_size = voxel_size
            self.gt_name = gt_name
            self.raw_name = raw_name
            if out_path is None:
                self.out_path = self.src
            else:
                self.out_path = out_path
            self.model_name = model_name
            self.model_path = model_path
            self.side_length = side_length
            self.unet_depth = unet_depth
            self.downsample_factor = downsample_factor
            self.conv_padding = conv_padding
            self.num_fmaps = num_fmaps
            self.fmap_inc_factor = fmap_inc_factor
            self.constant_upsample = constant_upsample
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.init_learning_rate = init_learning_rate
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

    def imshow(self, raw, gt=None, prediction=None, context=None):
        cols = 1 + (gt is not None) + (prediction is not None) + (context is not None)
        fig, axes = plt.subplots(1, cols, figsize=(30,30*cols))
        if len(raw.shape) == 3:
            middle = raw.shape[0] // 2
            axes[0].imshow(raw[middle], cmap='gray', vmin=0, vmax=1)
        else:
            axes[0].imshow(raw, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Raw')
        col = 1
        if gt is not None:
            if len(gt.shape) == 3:
                axes[col].imshow(gt[middle], vmin=0, vmax=1)
            else:
                axes[col].imshow(gt, vmin=0, vmax=1)
            axes[col].set_title('Ground Truth')
            col += 1
        if prediction is not None:
            if prediction.size < raw.size:
                pads = np.subtract(raw.shape, prediction.shape) // 2
                pad_tuple = []
                for p in pads:
                    pad_tuple.append((p, p))
                prediction = np.pad(prediction, tuple(pad_tuple))
            if len(prediction.shape) == 3:
                axes[col].imshow(prediction[middle], cmap='gray')
            else:
                axes[col].imshow(prediction, cmap='gray')
            axes[col].set_title('Prediction')
            col += 1
        if context is not None:
            middle = context.shape[0] // 2
            if len(context.shape) == 3:
                axes[col].imshow(context[middle], cmap='gray', vmin=0, vmax=1)
            else:
                axes[col].imshow(context, cmap='gray', vmin=0, vmax=1)
            axes[col].set_title('Context')
            col += 1
   
    def batch_show(self, batch=None, i=0):
        if batch is None:
            batch = self.batch
        self.imshow(batch[self.raw].data[i].crop(batch[self.gt].spec.roi).data[i].squeeze(), 
            batch[self.gt].data[i].squeeze(), 
            batch[self.prediction].data[i].squeeze(),
            batch[self.raw].data[i].squeeze(),
            )
    
    def batch_tBoard_write(self, i=0):
        for array in self.arrays:
            if array in self.crops.keys():
                img = self.batch[array].crop(self.batch[self.crops[array]].spec.roi).data[i].squeeze()
            else:
                img = self.batch[array].data[i].squeeze()
            mid = img.shape[0] // 2 # TODO: assumes 3D volume
            self.trainer.summary_writer.add_image(array.identifier, img[mid], global_step=self.trainer.iteration, dataformats='HW')

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
        self.raw = gp.ArrayKey('RAW') # raw data
        self.gt = gp.ArrayKey('GT') # ground truth data
        self.prediction = gp.ArrayKey('PREDICTION') # prediction of GT data from raw
        self.arrays = [self.raw, self.gt, self.prediction]
        
        # automatically generate some config variables
        self.gen_context_side_length()
        self.crops = {self.raw:self.prediction}
        
        # setup data source
        self.source = gp.ZarrSource(    # add the data source
            self.src,  # the zarr container
            {   self.raw: self.raw_name,
                self.gt: self.gt_name
                },  # which dataset to associate to the array key
            {   self.raw: gp.ArraySpec(interpolatable=True),
                self.gt: gp.ArraySpec(interpolatable=True)
                }  # meta-information
        )
        
        # get performance stats
        self.performance = gp.PrintProfilingStats(every=self.log_every)

        # stack for batches
        self.stack = gp.Stack(self.batch_size) # TODO: Determine if removing increases speed

        # setup a cache
        self.cache = gp.PreCache(num_workers=os.cpu_count())

        # define our network model for training
        self.build_model()
        
        # add normalization
        self.normalize_raw = gp.Normalize(self.raw)
        self.normalize_gt = gp.Normalize(self.gt)
        self.normalize_pred = gp.Normalize(self.prediction)
        
        # add a RandomLocation node to the pipeline to randomly select a sample
        self.random_location = gp.RandomLocation()

    def build_model(self):
        self.unet = UNet(
                in_channels=1,
                num_fmaps=self.num_fmaps,
                fmap_inc_factor=self.fmap_inc_factor,
                downsample_factors=[(self.downsample_factor,)*3,] * (self.unet_depth - 1),
                padding=self.conv_padding,
                constant_upsample=self.constant_upsample,
                voxel_size=self.voxel_size # set for each dataset
                )

        self.model = torch.nn.Sequential(
                            self.unet,
                            ConvPass(self.num_fmaps, 1, [(1, 1, 1)], activation=None),
                            torch.nn.Sigmoid())        

    def build_training_pipeline(self):
        # add transpositions/reflections
        self.simple_augment = gp.SimpleAugment()

        # prepare tensors for UNet
        unsqueeze = gp.Unsqueeze([self.raw, self.gt]) # context dependent so not added to object

        # pick loss function
        self.loss = torch.nn.MSELoss()

        # pick optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=self.init_learning_rate)

        # create a train node using our model, loss, and optimizer
        self.trainer = gp.torch.Train(
                            self.model,
                            self.loss,
                            self.optimizer,
                            inputs = {
                                'input': self.raw
                            },
                            loss_inputs = {
                                0: self.prediction,
                                1: self.gt
                            },
                            outputs = {
                                0: self.prediction
                            },
                            log_dir=self.tensorboard_path,
                            log_every=self.log_every,
                            checkpoint_basename=self.model_path+self.model_name,
                            save_every=self.save_every
                            )

        # create request
        self.train_request = gp.BatchRequest()
        self.train_request.add(self.raw, self.voxel_size*self.context_side_length)
        self.train_request.add(self.gt, self.voxel_size*self.side_length)
        self.train_request.add(self.prediction, self.voxel_size*self.side_length)

        # assemble pipeline
        self.training_pipeline = (self.source +
                                self.normalize_raw + 
                                self.normalize_gt +
                                self.random_location +
                                self.simple_augment + 
                                unsqueeze + 
                                self.cache +
                                self.stack + 
                                self.trainer +
                                self.performance
                                )
    
    def gen_context_side_length(self):
        # figure out proper ROI padding for context for the UNet generators
        self.conv_passes = 2 # set by default in unet
        self.kernel_size = 3 # set by default in unet
        self.context_side_length = 2 * np.sum([(self.conv_passes * (self.kernel_size - 1)) * (2 ** level) for level in np.arange(self.unet_depth - 1)]) + (self.conv_passes * (self.kernel_size - 1)) * (2 ** (self.unet_depth - 1)) + (self.conv_passes * (self.kernel_size - 1)) + self.side_length

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
            for i in range(self.num_epochs):
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
        scan = gp.Scan(scan_request, num_workers=self.batch_size)#os.cpu_count())

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


#===================================================================================

from gunpowder.torch import Train
import math
import sys
import functools
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

from tri_utils import NLayerDiscriminator3D, GANLoss, init_weights, UnetGenerator3D

n_samples = 11

l_gan_lambda = 2
l1_lambda = 150
l_gan_lambda = 1
l1_lambda = 100

data_dir = "/n/groups/htem/Segmentation/networks/tmn7/gunpowder/gt"
zarr_name = "cortex2.zarr"
zarr_path = os.path.join(data_dir, zarr_name)
log_dir = "logs"

# mult_xy = 0
# mult_xy = 4*30
mult_xy = 4*9
mult_xz = -4*21
# mult_xy = 0
# mult_xz = 0
input_shape = gp.Coordinate((132+mult_xz, 132+mult_xy, 132+mult_xy))
output_shape = gp.Coordinate((92+mult_xz, 92+mult_xy, 92+mult_xy))

voxel_size = gp.Coordinate((60, 60, 60))  # TODO: change later
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

checkpoint_every = 5000
train_until = 40000
snapshot_every = 1000
zarr_snapshot = False
num_workers = 11

net_g_num_fmaps = 64
net_d_num_fmaps = 64
net_g_num_fmaps = 32
# net_d_num_fmaps = 32

def mk_unet_3d():
    unet = UNet(
        in_channels=1,
        num_fmaps=net_g_num_fmaps,
        # num_fmaps_out=1,
        fmap_inc_factor=2,
        downsample_factors=[
            [2, 2, 2],
            [2, 2, 2],
            # [2, 2, 2],
            ],
        )

    return torch.nn.Sequential(
        unet,
        ConvPass(net_g_num_fmaps, 1, [[1, 1, 1]], activation='Sigmoid'))

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
    def __init__(self, l1_loss, gan_loss, netD, netG, optimizer_D, optimizer_G,
                 ):
        super(CycleGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = gan_loss
        self.netD = netD
        self.netG = netG
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

    def backward_D(self, fake_B, real_A, real_B):

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.gan_loss(pred_real, True)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        return loss_D

    def backward_G(self, fake_B, real_A, real_B):
        """Calculate GAN and L1 loss for the generator"""

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.gan_loss(pred_fake, True) * l_gan_lambda
        # Second, G(A) = B
        self.loss_G_L1 = self.l1_loss(fake_B, real_B) * l1_lambda  # TODO: check lambda
        # combine loss and calculate gradients
        loss_G = self.loss_G_GAN + self.loss_G_L1
        loss_G.backward()
        return loss_G

        # loss_G_L1 = self.l1_loss(fake_B, real_B) * 1  # TODO: check lambda
        # loss_G_L1.backward()
        # return loss_G_L1

    def forward(self, fake_B, real_A, real_B, mask):

        fake_B_mask = fake_B * mask
        real_A_mask = real_A * mask
        real_B_mask = real_B * mask

        # # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        loss_D = self.backward_D(fake_B_mask, real_A_mask, real_B_mask)
        self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        loss_G = self.backward_G(fake_B_mask, real_A_mask, real_B_mask)                   # calculate gradient for G
        self.optimizer_G.step()             # udpate G's weights

        self.loss_dict = {
            'loss_D': float(loss_D),
            'loss_G_GAN': float(self.loss_G_GAN),
            'loss_G_L1': float(self.loss_G_L1),
        }

        total_loss = loss_D + loss_G
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


def mknet(mask=None):
    norm_layer = functools.partial(torch.nn.InstanceNorm3d, affine=False, track_running_stats=False)

    netG = mk_unet_3d()
    # netG = UnetGenerator3D(input_nc=1, output_nc=1, num_downs=7, ngf=64, 
    #                      norm_layer=norm_layer, use_dropout=False)
    # init_weights(netG, init_type='orthogonal')
    init_weights(netG, init_type='normal', init_gain=0.05)
    # init_weights(netG, init_type='normal')
    # init_weights(netG, init_type='xavier')

    netD = NLayerDiscriminator3D(input_nc=2, ndf=net_d_num_fmaps, n_layers=3, norm_layer=norm_layer,
                                 downsampling_kw=2, kw=3,
                                 )
    init_weights(netD, init_type='normal')

    model = CycleGAN_Model(netG, netD)

    l1_loss = torch.nn.L1Loss()
    gan_loss = GANLoss(gan_mode='lsgan')

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.95, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.95, 0.999))
    optimizer = CycleGAN_Optimizer(optimizer_G, optimizer_D)

    loss = CycleGAN_Loss(l1_loss, gan_loss, netD, netG, optimizer_D, optimizer_G)

    return (model, loss, optimizer)


def train(iterations):

    source = gp.ArrayKey('source')
    source_cropped = gp.ArrayKey('source_cropped')
    target = gp.ArrayKey('target')
    fake = gp.ArrayKey('fake')
    mask = gp.ArrayKey('mask')
    gradients = gp.ArrayKey('GRADIENTS')

    model, loss, optimizer = mknet()

    # alias
    real_A = source
    real_A_cropped = source_cropped
    fake_B = fake
    real_B = target

    request = gp.BatchRequest()
    request.add(source, input_size)
    request.add(source_cropped, output_size)
    request.add(target, output_size)
    request.add(mask, output_size)
    request.add(gradients, output_size)

    snapshot_request = gp.BatchRequest()
    snapshot_request[fake] = request[target].copy()

    sources = tuple(
        [gp.ZarrSource(
            zarr_path,
            {
                source: f'volumes/raw_100nm',
                source_cropped: f'volumes/raw_100nm',
                target: f'volumes/raw_30nm',
                # mask: f'volumes/train_mask',
                mask: f'volumes/train_mask1',
                # mask: f'volumes/train_mask2',
            },
            {
                source: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                source_cropped: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                target: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                mask: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
            }) +
        gp.RandomLocation(min_masked=0.5, mask=mask) +
        # gp.RandomLocation(mask=mask) +
        # gp.RandomLocation() +
        # gp.Reject(mask=mask) +
        gp.Normalize(source) +
        gp.Normalize(source_cropped) +
        gp.Normalize(target)]
    )

    pipeline = sources
    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()
    pipeline += gp.ElasticAugment(
        # control_point_spacing=(64, 64),
        # control_point_spacing=(48*30, 48*30, 48*30),
        control_point_spacing=(48, 48, 48),
        jitter_sigma=(5.0, 5.0, 5.0),
        rotation_interval=(0, math.pi/2),
        subsample=4,
        )
    # pipeline += gp.IntensityAugment(
    #     source,
    #     scale_min=0.8,
    #     scale_max=1.2,
    #     shift_min=-0.2,
    #     shift_max=0.2)
    # pipeline += gp.NoiseAugment(source, var=0.01)
    # pipeline += gp.NoiseAugment(source, var=0.001)
    # pipeline += gp.NoiseAugment(source, var=0.002)

    # add "channel" dimensions
    pipeline += gp.Unsqueeze([source, target, source_cropped])
    # add "batch" dimensions
    pipeline += gp.Unsqueeze([source, target, source_cropped])
    # pipeline += gp.Unsqueeze([target])

    pipeline += gp.PreCache(num_workers=num_workers)
    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'real_A': source
            # 'x': source
            # 'input': source
        },
        outputs={
            0: fake_B,
        },
        gradients={
            0: gradients
         },
        loss_inputs={
            0: fake_B,
            1: real_A_cropped,
            2: real_B,
            3: mask,
        },
        log_dir = log_dir,
        save_every=checkpoint_every,
        )

    pipeline += gp.Squeeze([source, source_cropped, target, fake, gradients], axis=0)
    pipeline += gp.Squeeze([source, source_cropped, target, fake, gradients], axis=0)
    # pipeline += gp.Squeeze([gradients], axis=1)

    pipeline += gp.Snapshot({
            real_A: 'real_A',
            real_B: 'real_B',
            fake_B: 'fake_B',
            gradients: 'gradients',
            mask: 'mask',
        },
        every=snapshot_every,
        output_filename='batch_{iteration}.zarr' if zarr_snapshot else 'batch_{iteration}.hdf',
        additional_request=snapshot_request)

    with gp.build(pipeline):
        for i in tqdm(range(iterations)):
            pipeline.request_batch(request)

if __name__ == '__main__':

    import json
    mult_xy = 8*8
    # mult_xy = 8*8
    test_input_shape = gp.Coordinate((132+mult_xy, 132+mult_xy, 132+mult_xy))
    test_output_shape = gp.Coordinate((92+mult_xy, 92+mult_xy, 92+mult_xy))
    config = {
        'raw': 'real_A',
        'input_shape': test_input_shape,
        'output_shape': test_output_shape,
        'out_dims': 1,
        'out_dtype': "uint8",
    }
    with open('test_net.json', 'w') as f:
        json.dump(config, f)
    print('Dumped config...')
    if 'mknet' in sys.argv:
        exit()

    try:
        train_until = int(sys.argv[1])
    except:
        pass

    if 'test' in sys.argv:
        # global train_until
        train_until = 10
        snapshot_every = 1
        zarr_snapshot = True
        num_workers = 1

    train(train_until)
