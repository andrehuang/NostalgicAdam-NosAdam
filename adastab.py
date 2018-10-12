import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable


class AdaStab(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, gamma=0, lr_decay=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
       # if not 0.0 <= gamma:
       #     raise ValueError("Invalid gamma value: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, gamma=gamma, lr_decay=lr_decay)
        super(AdaStab, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaStab, self).__setstate__(state)
        for group in self.param_groups:
             group.setdefault('lr_decay', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                # amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['B_old'] = 0
                    state['B_new'] = 1
                    # if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        # state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                beta2 = state['B_old']/state['B_new']
                gamma = group['gamma']
                lr_decay = group['lr_decay']

                state['step'] += 1

                step = state['step']
                state['B_old'] += math.pow(step, -gamma)
                state['B_new'] += math.pow(step+1, -gamma)


                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad.mul_(grad))
                
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                # else:
#                 denom = exp_avg_sq.sqrt().add_(group['eps'])
#                 denom = np.cbrt(exp_avg_sq)
#                 denom = torch.Tensor(denom)
#                 denom = denom.cuda()
#                 denom = denom.add_(group['eps'])
                




                bias_correction1 = 1 - beta1 ** state['step']
                # AdaStab no longer needs bias correction for v_t
                # bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                if lr_decay:
                    step_size/math.sqrt(step)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
