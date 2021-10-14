#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""
Train models with score matching, anti score matching and gradient norm regularization.
"""

import argparse

import time
import sys
import os
from math import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.serialization import save
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from models.model_selector import model_architecture
from utils.misc_functions import create_folder

from regularizers import (GradNormRegularizer, 
                          ScoreMatching, 
                          AntiScoreMatching)

parser = argparse.ArgumentParser(description='Experiment arguments')

parser.add_argument( '-m','--model-name', default=None, 
                    help="Name of trained model to save, not necessary to specify")

parser.add_argument('--model-arch', default='resnet9', help='What architecture to use?')

parser.add_argument('--regularizer', choices=['score', 'anti', 'gnorm'],
                    help='Which regularizer to use? \
                        If no option is provided then no regularizer is used')

parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar100',
                    help='Which dataset to use?')

parser.add_argument('--regularization-constant', default=1e-5, type=float,
                    help='Value of regularization constant for optimization')

args = parser.parse_args()

# Choose datasets and number of classes
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

if args.cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

# Training hyper-parameters hard-coded for CIFAR10/100
training_args = {
    'batch_size': 128, 'epochs': 200, 'lr': 1e-1,
    'l2_val': 5e-4, 'gamma': 0.1, 'checkpoints':[100, 150]
    }


# Dataset loaders
train_loader = torch.utils.data.DataLoader(
        dataset_fn('.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()
                   ])),
    batch_size=training_args['batch_size'], shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset_fn('.', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=training_args['batch_size'], shuffle=False)


# Initialize model
model = model_architecture[args.model_arch](num_classes).to(device)

# Initialize optimizer and scheduler 
optimizer = optim.SGD(model.parameters(), lr=training_args['lr'], momentum=0.9, \
                            weight_decay=training_args['l2_val'])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_args['checkpoints'],\
                                        gamma=training_args['gamma'])


# Initialize regularization methods with hard-coded hyper-parameters
regularization_method = {
'gnorm': GradNormRegularizer(),
'score': ScoreMatching(model, num_samples=1, tau=1e-3, std_dev = 0.01),
'anti': AntiScoreMatching(model, num_samples=1, thresh=1e4, std_dev = 0.001),
}

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)
        batch_size = data.size(0)

        optimizer.zero_grad()
        model.zero_grad()

        output = model(data)
        loss_raw = F.nll_loss(F.log_softmax(output,1), target)

        if args.regularizer is not None:
            reg = regularization_method[args.regularizer].regularize(output, data, target)
            metadata = regularization_method[args.regularizer].metadata
            loss = loss_raw + args.regularization_constant * reg
        else:
            metadata = []
            loss = loss_raw 

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_raw.item()), end = '')

            for k in metadata:
                print('\t'+k+': {:.4f}'.format(metadata[k]), end='')
            print('\n', end='')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            output = F.log_softmax(out,1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc


for epoch in range(1, training_args['epochs'] + 1):
    save_path = 'saved_models/'
    create_folder(save_path)
    start = time.time()
    train(epoch)
    print("Time: {:.2f} s".format(time.time() - start))
    test()

    scheduler.step()    

    # Provide a model name based on the architecture and regularizer used to train the model
    if args.model_name is None:
        if args.regularizer is not None:
            args.model_name = save_path + args.model_arch + '_' + args.regularizer
        else:
            args.model_name = save_path + args.model_arch

    # Save the current model
    if epoch < training_args['epochs']:
        torch.save(model.cpu().state_dict(), args.model_name + '_' +str(epoch) + '.pt')
    else:
        torch.save(model.cpu().state_dict(), args.model_name + '.pt')

    # delete previously stored model  
    if epoch > 1:
        os.remove(args.model_name + '_' +str(epoch - 1) + '.pt')

    model.to(device)

