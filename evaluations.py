#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#


"""
Evaluate gradients and samples of learnt models
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils

from models.model_selector import model_architecture

import time
import sys
import os
from math import sqrt
import csv

from utils.grad import InputGradient
from utils.smoothgrad import SmoothGrad

from utils.sgld import SGLD

from utils.misc_functions import *
from utils.pixel_perturb import pixel_perturb

eval_methods = ['visualize-saliency-and-samples', 
                'compute-sample-quality', 
                'pixel-perturb']

parser = argparse.ArgumentParser(description='Arguments for Saliency metric')

# Optimization arguments
parser.add_argument( '-s','--saliency-method', 
                    choices=['loss_gradient','logit_gradient',
                    'proxylogit_gradient', 'proxyloss_gradient'],
                    default="logit_gradient", help="Choose the saliency method")

parser.add_argument( '-m','--model-name', 
                    default='saved_models/resnet9_cifar100.pt',
                    help="Choose the trained model")

parser.add_argument( '--model-arch', default='resnet18', 
                    choices=model_architecture.keys(),
                    help='Architecture of the model to evaluate')
    
parser.add_argument( '-pm','--proxy-model-name', 
                    default='saved_models/vgg11_cifar100.pt',
                    help="Choose the proxy model for saliency and sample quality computation")    

parser.add_argument( '--proxy-model-arch', default='resnet18', 
                    choices=model_architecture.keys(),
                    help='Architecture for the proxy model')

parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], 
                    default='cifar100',
                    help='Which dataset to use?')

parser.add_argument('--eval', choices=eval_methods, 
                    default = 'visualize-saliency-and-samples', 
                    help='which evaluation to perform')

parser.add_argument('--num-samples', type=int, default=1000, 
                    help='number of generated samples to calculate \
                    sample quality using GAN-test')

args = parser.parse_args()

if args.dataset == 'cifar100': 
    num_classes = 100
    dataset_fn = datasets.CIFAR100
else: 
    num_classes = 10
    dataset_fn = datasets.CIFAR10

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

if args.cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args.eval == 'visualize-saliency-and-samples': 
    batch_size = 9
else: batch_size = 256

train_loader = torch.utils.data.DataLoader(
        dataset_fn('.', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    dataset_fn('.', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=False)


model = model_architecture[args.model_arch](num_classes).to(device)
model.load_state_dict(torch.load(args.model_name))

proxy_model = model_architecture[args.proxy_model_arch](num_classes).to(device)
proxy_model.load_state_dict(torch.load(args.proxy_model_name))

# select saliency method    
saliency_obj = {
'proxylogit_gradient': InputGradient(proxy_model, loss=False),
'proxyloss_gradient': InputGradient(proxy_model, loss=True),
'loss_gradient': InputGradient(model, loss=True),
'logit_gradient': InputGradient(model, loss=False)
}

explanation = saliency_obj[args.saliency_method]

def check_accuracy(eval_model, input, target):
    output = eval_model(input)
    pred = output.data.max(1, keepdim=True)[1] 
    return pred.eq(target.data.view_as(pred)).cpu().float().sum().item()

def save_cifar_image(im, outfile):
    im = F.interpolate(im.detach(), size=64, mode='bilinear', align_corners=True)
    utils.save_image(im, outfile, nrow=3, scale_each=True, normalize=True)

def evaluate(method = 'visualize-saliency-and-samples', train = False):

    assert method in eval_methods

    save_path = 'results/' + method + '/'
    create_folder(save_path)

    model.eval()

    tl = train_loader if train else test_loader

    for idx, (data, target) in enumerate(tl):
        data, target = data.to(device).requires_grad_(), target.to(device)
        
        # Visualize Saliency and Samples
        if method == eval_methods[0]:

            cam = explanation.saliency(data, target)
            save_cifar_image(cam, args.outfile + 'saliency.png')
            save_cifar_image(data, args.outfile + 'image.png')         

            sgld = SGLD(model, num_steps=1000, lr=0.1, weight_decay=1e-2)
            sgld_denoise = SGLD(model, num_steps=200, lr=0.01, weight_decay=0.)
            
            samples = sgld.sample(target)
            save_cifar_image(samples, args.outfile + 'samples.png')

            noise = torch.normal(mean=torch.zeros_like(data).to(device), std=0.1)
            noisy_data = data + noise            
            denoised = sgld_denoise.sample(target, noisy_data)
            save_cifar_image(noisy_data, args.outfile + 'noisy_samples.png')
            save_cifar_image(denoised, args.outfile + 'denoised_samples.png')

        elif method == eval_methods[1]:

            if idx == 0:
                sgld = SGLD(model, num_steps=500, lr=0.1, weight_decay=1e-2)
                correct = 0
                correct_denoised = 0
                num_samples = 0
                proxy_model.eval()

            if num_samples < args.num_samples:
                samples = sgld.sample(target)
                correct += check_accuracy(proxy_model, samples, target)

                num_samples += data.size(0)
                print(correct / num_samples)
            else:
                correct /= num_samples
                print('GAN-test Score: ', correct)

        # Pixel Perturbation test
        elif method == eval_methods[2]:

            saliency_map = explanation.saliency(data, target_class=target)
            masking_fraction, accuracies = pixel_perturb(saliency_map, model, data, target)

            if idx == 0:
                evaluation_score = [masking_fraction, accuracies]
            else:
                for index, _ in enumerate(evaluation_score[1]):
                    evaluation_score[1][index] += accuracies[index]

    # Save results of pixel perturbation test
    if method == eval_methods[2]:
        for i, _ in enumerate(evaluation_score[1]):
            evaluation_score[1][i] /= len(tl.dataset)

        masking_fraction = evaluation_score[0]
        accuracies = evaluation_score[1]
        with open(save_path + args.model_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([masking_fraction, accuracies])


if __name__ == "__main__":
    print('Evaluation=', args.eval)
    evaluate(method = args.eval, train=False)



