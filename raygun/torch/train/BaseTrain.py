import inspect
from tqdm import trange
import gunpowder as gp

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
        for key, value in locals():
            setattr(self, key, value)
        
        self.arrays = {}
        for datapipe in datapipes.values():
            self.arrays.update(datapipe.arrays)

        self.input_dict = {}
        for array_name in inspect.signature(model.forward).parameters.keys():
            if array_name is not 'self':
                self.input_dict[array_name] = self.arrays[array_name]
                
        self.output_dict = {}
        for array_name in model.output_arrays:
            self.output_dict[array_name] = self.arrays[array_name]

        self.loss_input_dict = {}
        for array_name in inspect.signature(loss.forward).parameters.keys():
            if array_name is not 'self':
                self.loss_input_dict[array_name] = self.arrays[array_name]

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
        return (dp.prenet_pipe(mode) for dp in self.datapipes.values()) + gp.MergeProvider() #merge upstream pipelines for multiple sources
    
    def postnet_pipe(self):
        '''Implement in subclasses.'''
        raise NotImplementedError()
        
    def training_pipe(self, mode:str='train'):
        # assemble pipeline
        training_pipe = self.prenet_pipe(mode)
        training_pipe += self.train_node        
        training_pipe += self.postnet_pipe(mode)
        if mode == 'test':
            return training_pipe + gp.PrintProfilingStats(every=self.log_every)
        else:
            return training_pipe + gp.PreCache(num_workers=self.num_workers, cache_size=self.cache_size)
    
    def batch_tBoard_write(self):
        if hasattr(self.model, 'add_log'):
            self.model.add_log(self.train_node.summary_writer, self.train_node.iteration)
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
                if i % self.log_every == 0:
                    self.batch_tBoard_write()
        return self.batch
    
    def test(self, mode:str='train'):
        getattr(self.model, mode)() 
        training_pipeline = self.training_pipe(mode='test')
        with gp.build(training_pipeline):
            self.batch = training_pipeline.request_batch(self.batch_request)
        return self.batch
