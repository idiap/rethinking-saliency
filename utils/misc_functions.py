#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Misc helper functions """

import numpy as np
import subprocess

import torch
import torchvision.transforms as transforms

def maskify(cam, masking_fraction, type='largest-insensitive'):
    if isinstance(masking_fraction, float):
        return maskify_scalar(cam, masking_fraction, type)
    elif isinstance(maskify_scalar, torch.Tensor):
        return maskify_tensor(cam, masking_fraction, type)

def maskify_scalar(cam, masking_fraction = 0.2, type = 'largest-insensitive'):
    """
    Create a mask using a saliency map
    masking fraction -> fraction of image to mask out
    """

    # flatten
    grad_flat = cam.view((cam.size(0),-1))

    # add noise
    mean = torch.zeros_like(grad_flat).to(cam.device)
    grad_flat += torch.normal(mean, std = 1e-20)

    # Topk 
    # TODO: Differentiable TopK
    num_pixels_to_remove = int(masking_fraction * grad_flat.size(1))

    if type == 'largest-insensitive':
        topvalues, _ = torch.topk(grad_flat, k=num_pixels_to_remove , largest=False)
        leastval = topvalues[:, num_pixels_to_remove-1:]
        mask = torch.ge(grad_flat, leastval.detach()).float()
    elif type == 'smallest-sensitive':
        topvalues, _ = torch.topk(grad_flat, k=num_pixels_to_remove , largest=True)
        leastval = topvalues[:, num_pixels_to_remove-1:]
        mask = torch.le(grad_flat, leastval.detach()).float()
    else:
        print('Unknown option')

    # unflatten
    return mask.view_as(cam).detach()


def maskify_tensor(cam, masking_fraction, type = 'largest-insensitive'):
    """
    Create a mask using a saliency map
    masking fraction -> fraction of image to mask out (Tensor)
    """

    # flatten
    grad_flat = cam.view((cam.size(0),-1))

    # add noise
    mean = torch.zeros_like(grad_flat).to(cam.device)
    grad_flat += torch.normal(mean, std = 1e-20)

    num_pixels_to_remove = (masking_fraction * grad_flat.size(1)).long()

    if type == 'largest-insensitive':
        sorted_arr, _ = torch.sort(grad_flat, dim = 1, descending=False)
        leastval = torch.gather(sorted_arr, dim=1, index = num_pixels_to_remove)
        leastval = leastval.view((cam.size(0),1))
        mask = torch.ge(grad_flat, leastval).float()
    elif type == 'smallest-sensitive':
        sorted_arr, _ = torch.sort(grad_flat, dim = 1, descending=True)
        leastval = torch.gather(sorted_arr, dim=1, index = num_pixels_to_remove)
        leastval = leastval.view((cam.size(0),1))
        mask = torch.le(grad_flat, leastval).float()
    else:
        print('Unknown option')

    # unflatten
    return mask.view_as(cam).detach()


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None
