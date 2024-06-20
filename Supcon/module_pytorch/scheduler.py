#!/usr/bin/env python
# coding:utf-8
import math
import torch

class StepLR():
    def __init__(self, optimizer, step_size=20, gamma=0.5):
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)


class ExponentialLR():
    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

class CosineAnnealingLR():
    def __init__(self, optimizer, T_max=20, eta_min=0.001):
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

class ReduceLROnPlateau():
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True)

    def step(self, loss=0):
        self.lr_scheduler.step(loss)
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

class OneCycleLRScheduler():
    def __init__(self, optimizer, max_lr=0.3, steps_per_epoch=None, epochs=None):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer, 
                max_lr=self.max_lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs)

    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

# https://deecode.net/?p=1028
class ShelfNetLRSchedulerFunc:
    def __init__(self, base_lr, max_epoch=0, power=0.9):
        self._max_epoch = max_epoch
        self._power = power
        self._base_lr = base_lr

    def __call__(self, step=0):
        return (1 - max(step - 1, 1) / self._max_epoch) ** self._power * self._base_lr

class ShelfNetLRScheduler:
    def __init__(self, optimizer, base_lr, max_epoch=0, power=0.9):
        self.optimizer = optimizer
        lr_scheduler_func = ShelfNetLRSchedulerFunc(base_lr, max_epoch, power)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_func)
        
    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

# https://deecode.net/?p=1028
class CosineDecayLRSchedulerFunc:
    def __init__(self, max_epochs, warmup_lr_limit=0.16, warmup_epochs=5):
        self._max_epochs = max_epochs
        self._warmup_lr_limit = warmup_lr_limit
        self._warmup_epochs = warmup_epochs

    def __call__(self, step=0):
        step = max(step, 1)
        if step <= self._warmup_epochs:
            return self._warmup_lr_limit * step / self._warmup_epochs
        step -= 1
        rad = math.pi * step / self._max_epochs
        weight = (math.cos(rad) + 1.) / 2
        return self._warmup_lr_limit * weight

class CosineDecayLRScheduler:
    def __init__(self, optimizer, max_epochs, warmup_lr_limit=0.16, warmup_epochs=5):
        self.optimizer = optimizer
        lr_scheduler_func = CosineDecayLRSchedulerFunc(max_epochs, warmup_lr_limit, warmup_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_func)
        
    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

class CosineLRwithWarmupRLSchedulerFunc:
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles=7./16.):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

    def __call__(self, current_step=0):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        no_progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0., math.cos(math.pi * self.num_cycles * no_progress))

class CosineLRwithWarmupRLScheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
        self.optimizer = optimizer
        lr_scheduler_func = CosineLRwithWarmupRLSchedulerFunc(num_warmup_steps, num_training_steps, num_cycles)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler_func, last_epoch)

    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

class StepDecayRLSchedulerFunc:
    def __init__(self, base_lr, cycle):
        self.base_lr = base_lr
        self.cycle = cycle

    def __call__(self, step=0):
        lr_decay = self.base_lr * math.pow(0.5, step / self.cycle / 3.0)
        y =1.0 - math.fabs(step % self.cycle - self.cycle // 2) / (self.cycle / 2)
        x = lr_decay*0.1 + lr_decay*0.9*y
        return x

class StepDecayLRScheduler:
    def __init__(self, optimizer, base_lr, cycle):
        self.optimizer = optimizer
        lr_scheduler_func = StepDecayRLSchedulerFunc(base_lr, cycle)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_func)
        
    def step(self, loss=0):
        self.lr_scheduler.step()
        print("lr = {}".format(self.optimizer.param_groups[0]['lr']))

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self,state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

