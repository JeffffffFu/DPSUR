
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop



def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for id,group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()


        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm


        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()


        def step_dp(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)


        def step_dp_agd(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

    return DPOptimizerClass

DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)

def get_dp_optimizer(dataset_name,algortithm,lr,momentum,C_t,sigma,batch_size,model):

    if dataset_name=='IMDB' and algortithm!='DPAGD':
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
        )
    else:
        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
    return optimizer
