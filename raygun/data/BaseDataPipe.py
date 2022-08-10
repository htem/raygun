import gunpowder as gp

class BaseDataPipe(object):
    def __init__(self):
        pass
    
    def prenet_pipe(self, train=True):        
        # Make pre-net datapipe
        train_pipe = self.source + gp.RandomLocation()
        if train:
            sections = ['reject', 'resample', 'augment', 'unsqueeze', 'stack']
        else:
            sections = ['reject', 'resample', 'unsqueeze', 'stack']

        for section in sections:
            if hasattr(self, section) and getattr(self, section) is not None:
                train_pipe += getattr(self, section)
