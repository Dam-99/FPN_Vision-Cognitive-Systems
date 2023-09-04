import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import v2 as transforms_v2
from torchvision.datasets import wrap_dataset_for_transforms_v2

import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import time
import os
import re
# For results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# COCO
from pycocotools.coco import COCO

from helper import calculate_accuracy, model_training, plot_results, count_parameters, BATCH_SIZE

device = torch.device('cuda')

DS_TYPE = True # Used to have a smaller dataset to test things with
ROOT = './data'
COCO_DATA = 'train2017'
COCO_DATA_VAL = 'val2017'
COCO_DATA_TEST = 'test2017'
ROOT_TRAIN = f'{ROOT}/{COCO_DATA}'
ROOT_VAL = f'{ROOT}/{COCO_DATA_VAL}'
ROOT_TEST = f'{ROOT}/{COCO_DATA_TEST}'
ANN_TRAIN = f'{ROOT}/annotations/instances_{COCO_DATA}.json'
ANN_VAL = f'{ROOT}/annotations/instances_{COCO_DATA_VAL}.json'
ANN_TEST = f'{ROOT}/annotations/image_info_{COCO_DATA_TEST}.json'

BATCH_SIZE = 256
MODELS_PATH = 'models/'

if not (os.path.exists(ROOT) and os.path.exists(ANN)):
    print(f'COCO-{COCO_DATA} dataset not found')
    exit(2)
    
# def labels_getter(inputs):
#         return inputs[1]['labels']
    
train_transforms = transforms_v2.Compose([
                                            transforms_v2.ToImageTensor(),
                                            # transforms_v2.Resize((420, 640), antialias=True),
                                            # transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                            # transforms_v2.SanitizeBoundingBox(),
])
test_transforms = transforms_v2.Compose([
                                            transforms_v2.ToImageTensor(),
                                            # transforms_v2.Resize((420, 640), antialias=True),
                                            # transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                            # transforms_v2.SanitizeBoundingBox(),
])

if DS_TYPE:
        train_data = CocoDetection(
                root=ROOT_TRAIN,
                annFile=ANN_TRAIN,
                transforms=train_transforms
        )
        valid_data = CocoDetection(
                root=ROOT_VAL,
                annFile=ANN_VAL,
                transforms=test_transforms
        )
# TOFIX: this should be splitting the training dataset instead of taking the validation one two times, but typings later on don't work
else:
        train_data = CocoDetection(
                root=ROOT_VAL,
                annFile=ANN_VAL,
                transforms=train_transforms
        )
        valid_data = CocoDetection(
                root=ROOT_VAL,
                annFile=ANN_VAL,
                transforms=test_transforms
        )
    # num_train_examples = int(len(train_data) * 0.8)
    # num_valid_examples = len(train_data) - num_train_examples
    # print('Splitting val for training:', num_train_examples, num_valid_examples)

    # train_data, valid_data = data.random_split(train_data, [num_train_examples, num_valid_examples])
    # valid_data = copy.deepcopy(valid_data) # changing train transformations won't affect the validation set
    # valid_data.dataset.transform = test_transforms

test_data = CocoDetection(
        root=ROOT_TEST,
        annFile=ANN_TEST,
        transforms=test_transforms
)

print('TRAIN:', train_data)
print('VALID:', valid_data)
print('TEST:', test_data)

# Wrapper for better handling of data in Object detection
train_data = wrap_dataset_for_transforms_v2(train_data)
valid_data = wrap_dataset_for_transforms_v2(valid_data)
test_data = wrap_dataset_for_transforms_v2(test_data)

train_iterator = data.DataLoader(train_data,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE
)
valid_iterator = data.DataLoader(valid_data,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE
)
test_iterator = data.DataLoader(test_data,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE
)

resnet = torchvision.models.resnet

resnet50 = resnet.resnet50(weights='DEFAULT')
# print(resnet50)
resnet50 = resnet50.to(device)
# resnet101 = resnet.resnet101(weights='DEFAULT')
# resnet101 = resnet101.to(device)
#? layer_i is the same as conv_i+1 in the paper

#? register_forward_hook maybe for adding the new outputs

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
    
    def _lateral(self, C_i, P_i_plus_1=nn.Identity(), sum_components=True): #* merge operation of C_i with upsampled P_i-1
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
            layers[name] =  {"level": int(re.search(name, "\d").group()), "module": subm}
            if layers[name].level > last:
                last = layers[name].level
        for name in sorted(layers.keys(), reverse=True):
            l = layers[name].level
            m = layers[name].module
            if l == last:
                self._lateral(m, sum_components=False)
            else:
                self._lateral(m, getattr(self, f'p{l+1}'))

model = RPN(resnet50)
# print(model)
print(f"The model has {count_parameters(model):,} trainable parameters")

#! wrong criterion for object classification
criterion = nn.CrossEntropyLoss() # softmax + crossentropy
criterion = criterion.to(device)

optimizer = optim.SGD(model.parameters(), lr=3e-3) # could be anything, like adam
model = model.to(device)

N_EPOCHS = 25
train_losses, train_accs, valid_losses, valid_accs = model_training(N_EPOCHS, 
                                                                    model, 
                                                                    train_iterator, 
                                                                    valid_iterator, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    device,
                                                                    'rpm.pt')

plot_results(N_EPOCHS, train_losses, train_accs, valid_losses, valid_accs)