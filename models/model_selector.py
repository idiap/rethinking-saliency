#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

from models.resnet import *
from models.vgg import *

model_architecture = {
'vgg5': vgg5,
'vgg7': vgg7,
'vgg11': vgg11,
'vgg14': vgg14,
'resnet9': resnet9,
'resnet18': resnet18,
'resnet34': resnet34,
'resnet50': resnet50,
}