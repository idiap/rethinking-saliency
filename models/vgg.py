#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

beta = 10

cfg = {
    '5'  : [64,     'M', 128,      'M', 256,                'M', 512,                'M',                     ],
    '7'  : [64,     'M', 128,      'M', 256, 256,           'M', 512,                'M', 512,                ],
    '9' :  [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           ],
    '11' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           ],
    '14' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      ],
    '17' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, ]
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Linear(512*4, num_classes)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.Softplus(beta=10)]
        input_channel = l
    
    return nn.Sequential(*layers)


def vgg5(num_classes=100):
    return VGG(make_layers(cfg['5'], batch_norm=True), num_classes = num_classes)

def vgg7(num_classes=100):
    return VGG(make_layers(cfg['7'], batch_norm=True), num_classes = num_classes)

def vgg9(num_classes=100):
    return VGG(make_layers(cfg['9'], batch_norm=True), num_classes = num_classes)

def vgg11(num_classes=100):
    return VGG(make_layers(cfg['11'], batch_norm=True), num_classes = num_classes)

def vgg14(num_classes=100):
    return VGG(make_layers(cfg['14'], batch_norm=True), num_classes = num_classes)

def vgg17(num_classes=100):
    return VGG(make_layers(cfg['17'], batch_norm=True), num_classes = num_classes)


