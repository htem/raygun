import gunpowder as gp

class BaseDataPipe(object):
    def __init__(self):
        pass
    
    def prenet_pipe(self, mode:str='train'):        
        # Make pre-net datapipe
        prenet_pipe = self.source + gp.RandomLocation()
        if mode == 'train':
            sections = ['reject', 'resample', 'preprocess', 'augment', 'unsqueeze', 'stack']
        elif mode == 'predict':
            sections = ['reject', 'resample', 'preprocess', 'unsqueeze', 'stack']
        elif mode == 'test':
            sections = ['reject', 'resample', 'preprocess', 'unsqueeze', gp.Stack(1)]
        else:
            raise ValueError(f'mode={mode} not implemented.')

        for section in sections:
            if isinstance(section, str) and hasattr(self, section) and getattr(self, section) is not None:
                prenet_pipe += getattr(self, section)
            else:
                prenet_pipe += section
        
        return prenet_pipe
