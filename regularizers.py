#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""
Regularizers to use for training model
"""

import torch
import torch.nn.functional as F

class Regularizer():
    def __init__(self):
        self.metadata = {}

    def regularize(self, out, input, target):
        raise NotImplementedError()


class GradNormRegularizer(Regularizer):
    # norm of gradient of out w.r.t. input
    # usually out are logits or log-probabilites

    def __init__(self):
        super().__init__()

    def regularize(self, out, input, target):
        scalar_out = -1. * F.nll_loss(out, target, reduction='sum')
        grad_x = torch.autograd.grad(scalar_out, input, only_inputs=True, create_graph= True)[0]
        gradnorm = grad_x.pow(2).sum() / input.size(0)

        self.metadata = {
            'gradnorm': gradnorm.item()
        }

        return 0.5 * gradnorm


class ScoreMatching(Regularizer):
    def __init__(self, model, num_samples = 10, std_dev = 0.01, tau=0):
        super().__init__()
        """
        Regularized score matching: E_p(x) (trace(hessian) + 0.5 * grad-norm**2 + tau * trace(hessian)^2)
        Taylor trick: trace(hessian) = 2/(\sigma)^2 E_v(f(x+v) - f(x))
        
        num_samples -> used for monte carlo approximation of E_v
        std_dev -> gaussian noise variable for tr(H) computation
        tau -> reg constant for stability regularizer t(H)^2
        """

        self.model = model
        self.num_samples = num_samples
        self.std_dev = std_dev
        self.tau = tau

    def regularize(self, out, input, target):
        agg = -1. * F.nll_loss(out, target, reduction='sum')
        grad_x = torch.autograd.grad(agg, input, only_inputs=True, create_graph= True)[0]

        trace_hessian = 0.
        for i in range(self.num_samples):
            noise = torch.normal(mean=torch.zeros_like(input).to(input.device), std=self.std_dev)
            trH = self.model(input + noise) - out
            temp = (2 / self.std_dev**2 ) * -1. * F.nll_loss(trH, target, reduction='none')  
            trace_hessian += temp / self.num_samples

        # Mean across the input batch
        trace_hessian_sq = (trace_hessian**2).mean()
        trace_hessian = trace_hessian.mean()
        gradnorm = grad_x.pow(2).sum() / input.size(0)

        self.metadata = {
            'trace_hessian': trace_hessian.item(),
            'gradnorm': gradnorm.item() 
        }

        return trace_hessian + 0.5 * gradnorm + self.tau * (trace_hessian_sq)


class AntiScoreMatching(Regularizer):
    # Negative thresholded Hessian-trace
    # When minimized, increases the Hessian-trace 
    
    def __init__(self, model, num_samples = 1, std_dev = 0.01, thresh=1e4):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.std_dev = std_dev
        self.thresh = thresh

    def regularize(self, out, input, target):
        trace_hessian = 0
        for i in range(self.num_samples):
            noise = torch.normal(mean=torch.zeros_like(input).to(input.device), std=self.std_dev)            
            trH = self.model(input + noise) - out
            temp = (2 / self.std_dev**2 ) * -1. * F.nll_loss(trH, target, reduction='none') 
            trace_hessian += temp / self.num_samples

        trace_hessian = torch.clamp(trace_hessian, max=self.thresh).mean()

        self.metadata = {
            'trace_hessian': trace_hessian.item()
        }

        return -1. * trace_hessian 


