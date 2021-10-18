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
logger = logging.Logger('CARE', 'INFO')

class CARE():
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
            self.ndims = sum(voxel_size == np.min(voxel_size))
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
            if len(img.shape) == 3:
                mid = img.shape[0] // 2
                img = img[mid]
            self.trainer.summary_writer.add_image(array.identifier, img, global_step=self.trainer.iteration, dataformats='HW')

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
                downsample_factors=[(self.downsample_factor,)*self.ndims,] * (self.unet_depth - 1), 
                padding=self.conv_padding,
                constant_upsample=self.constant_upsample,
                voxel_size=self.voxel_size # set for each dataset
                )

        self.model = torch.nn.Sequential(
                            self.unet,
                            ConvPass(self.num_fmaps, 1, [(1,)*self.ndims], activation=None), 
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
        # figure out proper ROI padding for context
        self.conv_passes = 2 # set by default in unet
        self.kernel_size = 3 # set by default in unet
        if self.conv_padding == 'valid':
            self.context_side_length = 2 * np.sum([(self.conv_passes * (self.kernel_size - 1)) * (2 ** level) for level in np.arange(self.unet_depth - 1)]) + (self.conv_passes * (self.kernel_size - 1)) * (2 ** (self.unet_depth - 1)) + (self.conv_passes * (self.kernel_size - 1)) + self.side_length
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

