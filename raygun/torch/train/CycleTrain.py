import gunpowder as gp

from raygun.torch.train import BaseTrain

class CycleTrain(BaseTrain):
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
        super().__init__(**locals()) #TODO: May not work

    def postnet_pipe(self, mode:str='train'):
        if mode == 'test':
            stack = lambda dp: 1
        else:
            stack = lambda dp: dp.batch_size

        return (dp.postnet_pipe(batch_size=stack(dp)) for dp in self.datapipes.values())
    