import inspect
from tqdm import trange
import gunpowder as gp
import numpy as np

from raygun.utils import passing_locals

class BaseTrain(object):
    def __init__(self,
                datapipes:dict, 
                batch_request:gp.BatchRequest,
                model, 
                loss, 
                optimizer, 
                tensorboard_path:str='./tensorboard/',
                log_every:int=20,
                checkpoint_basename:str='./models/model',
                save_every:int=2000,
                spawn_subprocess:bool=False,
                num_workers:int=11,
                cache_size:int=50,
                **kwargs
                ):
        kwargs = passing_locals(locals())
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.arrays = {}
        for datapipe in datapipes.values():
            self.arrays.update(datapipe.arrays)

        self.input_dict = {}
        for array_name in inspect.signature(model.forward).parameters.keys():
            if array_name is not 'self':
                self.input_dict[array_name] = self.arrays[array_name]
                
        self.output_dict = {}
        for i, array_name in enumerate(model.output_arrays):
            if array_name not in self.arrays.keys():
                self.arrays[array_name] = gp.ArrayKey(array_name.upper())
            self.output_dict[i] = self.arrays[array_name]

        self.loss_input_dict = {}
        for i, array_name in enumerate(inspect.signature(loss.forward).parameters.keys()):
            if array_name is not 'self':
                self.loss_input_dict[i] = self.arrays[array_name]

        # create a train node using our model, loss, and optimizer
        self.train_node = gp.torch.Train(
                            model,
                            loss,
                            optimizer,
                            inputs = self.input_dict,
                            outputs = self.output_dict,
                            loss_inputs = self.loss_input_dict,
                            log_dir=tensorboard_path,
                            log_every=log_every,
                            checkpoint_basename=checkpoint_basename,
                            save_every=save_every,
                            spawn_subprocess=spawn_subprocess
                            )

    def prenet_pipe(self, mode:str='train'):        
        return tuple([dp.prenet_pipe(mode) for dp in self.datapipes.values()]) + gp.MergeProvider() #merge upstream pipelines for multiple sources
    
    def postnet_pipe(self):
        '''Implement in subclasses.'''
        raise NotImplementedError()
        
    def training_pipe(self, mode:str='train'):
        # assemble pipeline
        training_pipe = self.prenet_pipe(mode)
        training_pipe += self.train_node        
        for section in self.postnet_pipe(mode):
            training_pipe += section
        if mode == 'test':
            return training_pipe + gp.PrintProfilingStats(every=self.log_every)
        else:
            return training_pipe + gp.PreCache(num_workers=self.num_workers, cache_size=self.cache_size)
    
    def batch_tBoard_write(self):#TODO: This will break if spawn_subprocess==True
        if hasattr(self.model, 'add_log'):
            self.model.add_log(self.train_node.summary_writer, self.train_node.iteration)        

        if hasattr(self.loss, 'add_log'):
            self.loss.add_log(self.train_node.summary_writer, self.train_node.iteration)
        
        for array in self.train_node.loss_inputs.values():
                if len(self.batch[array].data.shape) > 3: # pull out self.batch dimension if necessary
                    img = self.batch[array].data[0].squeeze()
                else:
                    img = self.batch[array].data.squeeze()
                if len(img.shape) == 3:
                    mid = img.shape[0] // 2 # for 3D volume
                    data = img[mid]
                else:
                    data = img
                if (data.dtype == np.float32) and (data.min() < 0) and (data.min() >= -1.) and (data.max() <= 1.): # scale data to [0,1] if necessary
                    data = (data * 0.5) + 0.5
                self.train_node.summary_writer.add_image(array.identifier, data, global_step=self.batch.iteration, dataformats='HW')

        self.train_node.summary_writer.flush()
        self.n_iter = self.train_node.iteration

    def train(self, iter:int):
        self.model.train()
        training_pipeline = self.training_pipe()
        with gp.build(training_pipeline):
            pbar = trange(iter)
            for i in pbar:
                self.batch = training_pipeline.request_batch(self.batch_request)

                if hasattr(self.loss, 'loss_dict'):
                    pbar.set_postfix(self.loss.loss_dict)

                if hasattr(self.model, 'update_status'):
                    self.model.update_status(self.train_node.iteration)
                                
                if hasattr(self.loss, 'update_status'):
                    self.loss.update_status(self.train_node.iteration)

                if i % self.log_every == 0:
                    self.batch_tBoard_write()
                    
        return self.batch
    
    def test(self, mode:str='train'):
        getattr(self.model, mode)() 
        training_pipeline = self.training_pipe(mode='test')
        with gp.build(training_pipeline):
            self.batch = training_pipeline.request_batch(self.batch_request)
        return self.batch
