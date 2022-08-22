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
            if array_name != 'self':
                self.input_dict[array_name] = self.arrays[array_name]
                
        self.output_dict = {}
        for i, array_name in enumerate(model.output_arrays):
            if array_name not in self.arrays.keys():
                self.arrays[array_name] = gp.ArrayKey(array_name.upper())
            self.output_dict[i] = self.arrays[array_name]

        self.loss_input_dict = {}
        for i, array_name in enumerate(inspect.signature(loss.forward).parameters.keys()):
            if array_name != 'self':
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
        if mode == 'train':
            training_pipe += gp.PreCache(num_workers=self.num_workers, cache_size=self.cache_size)
        training_pipe += self.train_node        
        for section in self.postnet_pipe(mode):
            training_pipe += section
        if mode == 'test':
            training_pipe += gp.PrintProfilingStats() #TODO: Figure out why this doesn't print / print from batch.profiling_stats

        return training_pipe

    def print_profiling_stats(self):
        stats = "\n"
        stats += "Profiling Stats\n"
        stats += "===============\n"
        stats += "\n"
        stats += "NODE".ljust(20)
        stats += "METHOD".ljust(10)
        stats += "COUNTS".ljust(10)
        stats += "MIN".ljust(10)
        stats += "MAX".ljust(10)
        stats += "MEAN".ljust(10)
        stats += "MEDIAN".ljust(10)
        stats += "\n"

        summaries = list(self.batch.profiling_stats.get_timing_summaries().items())
        summaries.sort()

        for (node_name, method_name), summary in summaries:

            if summary.counts() > 0:
                stats += node_name[:19].ljust(20)
                stats += method_name[:19].ljust(10) if method_name is not None else ' '*10
                stats += ("%d"%summary.counts())[:9].ljust(10)
                stats += ("%.2f"%summary.min())[:9].ljust(10)
                stats += ("%.2f"%summary.max())[:9].ljust(10)
                stats += ("%.2f"%summary.mean())[:9].ljust(10)
                stats += ("%.2f"%summary.median())[:9].ljust(10)
                stats += "\n"

        stats += "\n"

        print(stats)

    def train(self, iter:int):
        self.model.train()
        training_pipeline = self.training_pipe()
        with gp.build(training_pipeline):
            pbar = trange(iter)
            for i in pbar:
                self.batch = training_pipeline.request_batch(self.batch_request)
                pbar.set_postfix({'loss': self.batch.loss})
                self.n_iter = self.train_node.iteration

                if i % self.log_every == 0:
                    self.train_node.summary_writer.flush()
    
    def test(self, mode:str='train'):
        getattr(self.model, mode)() 
        training_pipeline = self.training_pipe(mode='test')
        with gp.build(training_pipeline):
            self.batch = training_pipeline.request_batch(self.batch_request)
            
        return self.batch
