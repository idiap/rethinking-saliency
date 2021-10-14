#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

import torch
import torch.nn.functional as F
import numpy as np
import copy 
from utils.misc_functions import maskify

def pixel_perturb(saliency_map, model, image, target, constant_val=0.5):
    """
        Pixel perturbation test

        constant_val: value to replace removed pixels by
    """
    model.eval()    
    masking_fraction = np.logspace(-2, 0, num=25)
    accuracy = []
    auc = 0

    for m in masking_fraction:
        mask = maskify( (saliency_map).std(1, keepdim=True), m, 'largest-insensitive')
        x_perturbed = image * mask + constant_val * (1- mask)
        y_perturbed = model(x_perturbed)
        pred = y_perturbed.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        temp = pred.eq(target.data.view_as(pred)).cpu().float().sum().item()
        accuracy.append(temp)

    return masking_fraction, accuracy
