import itertools
import torch

class BaseDummyOptimizer(torch.nn.Module):
    def __init__(self, scheduler=None, scheduler_kwargs={}, **optimizers):
        super().__init__()

        if scheduler is not None:
            self.schedulers = {}

        for name, optimizer in optimizers.items():
            setattr(self, name, optimizer)
        
            """
            For 'LambdaLR', we keep the same learning rate for the first <n_epochs> epochs
            and linearly decay the rate to zero over the next <n_epochs_decay> epochs.
            For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
            See https://pytorch.org/docs/stable/optim.html for more details.
            """

            if isinstance(scheduler, str):
                if scheduler == 'LambdaLR':
                    def lambda_rule(epoch):
                        lr_l = 1.0 - max(0, epoch + scheduler_kwargs['epoch_count'] - scheduler_kwargs['n_epochs']) / (scheduler_kwargs['n_epochs_decay'] + 1.0)
                        return lr_l
                    self.schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

                else:
                    self.schedulers[name] = getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **scheduler_kwargs)

            elif scheduler is not None:
                self.schedulers[name] = scheduler(optimizer, **scheduler_kwargs)
            else:
                self.schedulers[name] = scheduler

    def step(self):
        if self.schedulers is not None:
            for name, scheduler in self.schedulers.items():
                scheduler.step()
        