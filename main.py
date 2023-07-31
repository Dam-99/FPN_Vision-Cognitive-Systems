import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import time
import os
# For results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# COCO
from pycocotools.coco import COCO

device = torch.device('cuda')

ROOT = './data'
COCO_DATA = 'val2017'
ANN = f'{ROOT}/annotatins/instances_{COCO_DATA}.json'
DS_TYPE = False

if DS_TYPE:
    if not (os.path.exists(ROOT) and os.path.exists(ANN)):
        print(f'COCO-{COCO_DATA} dataset not found')
        exit(2)

    train_data = torchvision.datasets.CocoDetection(
        root=ROOT,
        annFile=ANN
        # transform=train_transform
    )
else:
    train_data = torchvision.datasets.CIFAR10(
        root=ROOT,
        train=True,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=ROOT,
        train=False,
        download=True
    )

print(train_data)
print(train_data.data.shape)
print(test_data)
print(test_data.data.shape)

def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)
import re

resnet = torchvision.models.resnet

resnet50 = resnet.resnet50(weights='DEFAULT')
# print(resnet50)
resnet50 = resnet50.cuda()

class RPN(nn.Module):
    def __init__(self, backbone: nn.Module, d=256) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self._d = d
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.p2 = nn.Conv2d(in_channels=self._d, out_channels=self._d, kernel_size=3) #* these are the 3x3 maps added after each merge
        self.p3 = nn.Conv2d(in_channels=self._d, out_channels=self._d, kernel_size=3)
        self.p4 = nn.Conv2d(in_channels=self._d, out_channels=self._d, kernel_size=3)
        self.p5 = nn.Conv2d(in_channels=self._d, out_channels=self._d, kernel_size=3)
    
    def _lateral(self, C_i, P_i_plus_1, sum_components=True): #* merge operation of C_i with upsampled P_i-1
        reduced_channels_C = nn.Conv2d(in_channels=self._d, out_channels=self._d, kernel_size=1)(C_i)
        if sum_components:
            upscaled_P = nn.Upsample(scale_factor=2, mode='nearest')(P_i_plus_1)
            return torch.add(upscaled_P, reduced_channels_C)
        else:
            return reduced_channels_C
    
    def forward(self, x):
        batch_size = x[0]
        print(type(x))
        x = self.backbone.forward(x)
        layers = {}
        last = 0
        for name, subm in self.backbone.named_children():
            if not name.find('layer'):
                pass
            print(f'Copying layer {name}')
            layers[name] =  {"level": int(re.search(name).group()), "module": subm}
            if layers[name].level > last:
                last = layers[name].level
        for name in sorted(layers.keys(), reverse=True):
            l = layers[name].level
            m = layers[name].module
            if l == last:
                self._lateral(m, sum_components=False)
            else:
                self._lateral(m, self[f'p{l+1}'])

