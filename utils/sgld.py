#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class SGLD():
    def __init__(self, model, num_steps=10, lr=1e-3, weight_decay=1e-3):
        self.model = model

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.device = m.weight.device
                break
        
        self.num_steps = int(num_steps)
        self.lr = lr
        self.weight_decay = weight_decay


    def sample(self, targets=None, init_x=None):
        
        if targets is None:
            targets = torch.randint(low=0, high=99, size=(9,))
        
        if init_x is None:
            init_x = torch.normal(mean=0.5*torch.ones((targets.size(0),3,32,32)).to(self.device), std=0.2)
            init_x = torch.clamp(init_x, 0, 1)
        
        x = init_x
        
        for i in range(1,self.num_steps+1):
            self.model.zero_grad()
            x = x.requires_grad_()

            out = self.model(x)
            agg = -1. * F.nll_loss(out, targets, reduction='sum')
            grad_x = torch.autograd.grad(agg, x, only_inputs=True)[0]

            with torch.no_grad():
                x = torch.clamp(x + (self.lr * (grad_x - self.weight_decay * (x - 0.5) )) , 0 ,1)

        return x



class ActivationMaximization():
    def __init__(self, model, num_steps=10, lr=1e-3, weight_decay=1e-3):
        self.model = model

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.device = m.weight.device
                break
        
        self.num_steps = int(num_steps)
        self.lr = lr
        self.weight_decay = weight_decay


    def sample(self, targets=None, init_x=None):
        
        if targets is None:
            targets = torch.randint(low=0, high=99, size=(9,))
        
        if init_x is None:
            init_x = torch.normal(mean=0.5*torch.ones((targets.size(0),3,32,32)).to(self.device), std=0.2)
            init_x = torch.clamp(init_x, 0, 1)
            
        x = init_x
        optimizer = torch.optim.SGD([x], lr=self.lr, momentum=0,
                            weight_decay=0.)
        x = x.requires_grad_()

        for i in range(1,self.num_steps+1):
            self.model.zero_grad()
            optimizer.zero_grad()
            
            out = self.model(x.clamp(0,1))
            loss = F.nll_loss(out, targets, reduction='sum') + \
                0.5 * self.weight_decay * (x - 0.5).pow(2).sum()
            
            loss.backward(inputs=[x])
            optimizer.step()
            
        return x.clamp(0,1)


