import torch

class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Chainable linear warmup scheduler."""

    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [lr * self.last_epoch / self.warmup_steps
                for lr in self.base_lrs]
