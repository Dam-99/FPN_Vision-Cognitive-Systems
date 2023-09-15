import json
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

from helper import calculate_accuracy, model_training, plot_results, count_parameters, BATCH_SIZE, cache_empty, find_free_anns_ids, repair_empty, BROKEN_PATH

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

if not (os.path.exists(ROOT) and os.path.exists(ANN_TRAIN)):
    print(f'COCO-{COCO_DATA} dataset not found')
    exit(2)
    
def labels_getter(inputs):
        return inputs[1]['labels']

#? see FPN's _lateral method for disclaimer
train_transforms = transforms_v2.Compose([
                                            transforms_v2.ToImageTensor(),
                                            transforms_v2.Resize((640, 640), antialias=True),
                                            transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                            # transforms_v2.SanitizeBoundingBox(),
])
test_transforms = transforms_v2.Compose([
                                            transforms_v2.ToImageTensor(),
                                            transforms_v2.Resize((640, 640), antialias=True),
                                            transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                            # transforms_v2.SanitizeBoundingBox(),
])

def load_data():
    if DS_TYPE:
            train_data = CocoDetection(
                    root=ROOT_TRAIN,
                    annFile=ANN_TRAIN,
                    transforms=train_transforms
            )
            val_data = CocoDetection(
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
            val_data = CocoDetection(
                    root=ROOT_VAL,
                    annFile=ANN_VAL,
                    transforms=test_transforms
            )
        # num_train_examples = int(len(train_data) * 0.8)
        # num_valid_examples = len(train_data) - num_train_examples
        # print('Splitting val for training:', num_train_examples, num_valid_examples)

        # train_data, val_data = data.random_split(train_data, [num_train_examples, num_valid_examples])
        # val_data = copy.deepcopy(val_data) # changing train transformations won't affect the validation set
        # val_data.dataset.transform = test_transforms

    test_data = CocoDetection(
            root=ROOT_TEST,
            annFile=ANN_TEST,
            transforms=test_transforms
    )
        
    # Wrapper for better handling of data in Object detection
    train_data = wrap_dataset_for_transforms_v2(train_data)
    val_data = wrap_dataset_for_transforms_v2(val_data)
    test_data = wrap_dataset_for_transforms_v2(test_data)
    
    return train_data, val_data, test_data

train_data, val_data, test_data = load_data()

print('TRAIN:', train_data)
print('VALID:', val_data)
print('TEST:', test_data)

#? needed even if images have the same size because of the targets being different
def collate_fn(batch): # TODO: could be improved for less bloated code in training and model
    return tuple(zip(*batch))

def create_iterators(train_data=train_data, val_data=val_data, test_data=test_data):
    train_iterator = data.DataLoader(train_data,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    batch_size=BATCH_SIZE,
                                    # num_workers=1
    )
    val_iterator = data.DataLoader(val_data,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    batch_size=BATCH_SIZE,
                                    # num_workers=1
    )
    test_iterator = data.DataLoader(test_data,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    batch_size=BATCH_SIZE,
                                    # num_workers=1
    )
    return train_iterator, val_iterator, test_iterator

train_iterator, val_iterator, test_iterator = create_iterators()
     
def check_empty_targets(iterator, dataset, use_cache, ds_name, ann_ids):
  try:
    # iterator will try to create batches before the loop starts, so the error will be raised right away
    for x,y in iterator:
        break
  except:
    repair_empty(dataset, ds_name, use_cache, ann_ids)
    global train_data, val_data, test_data
    train_data, val_data, test_data = load_data()
    global train_iterator, val_iterator, test_iterator
    train_iterator, val_iterator, test_iterator = create_iterators()

missing_train_ann_ids = []
train_anns_sorted = sorted(train_data.coco.getAnnIds())
with open(F'{BROKEN_PATH}train.json', 'r') as f:
    train_missing = len(json.load(f))
find_free_anns_ids(train_anns_sorted, missing_train_ann_ids, train_missing)

missing_val_ann_ids = []
val_anns_sorted = sorted(val_data.coco.getAnnIds())
with open(f'{BROKEN_PATH}val.json', 'r') as f:
    val_missing = len(json.load(f))
find_free_anns_ids(val_anns_sorted, missing_val_ann_ids, val_missing)

cache = False
use_cache = True
check = True

#! These functions won't work if the "SanitizeBoundingBoxes" transforms is active, and other transforms may ruin the result
if cache:
    print('Checking Train Dataset...')
    cache_empty(train_data, 'train')
    print('Checking Valid Dataset...')
    cache_empty(val_data, 'val')
    print('Checking Test Dataset...')

if check:
    print('Checking Train Dataset...')
    check_empty_targets(train_iterator, train_data, use_cache, 'train', missing_train_ann_ids)
    print('Checking Valid Dataset...')
    check_empty_targets(val_iterator, val_data, use_cache, 'val', missing_val_ann_ids)
    print('Done')


resnet = torchvision.models.resnet

resnet50 = resnet.resnet50(weights='DEFAULT')
# print(resnet50)
resnet50 = resnet50.to(device)
# resnet101 = resnet.resnet101(weights='DEFAULT')
# resnet101 = resnet101.to(device)
#? layer_i is the same as conv_i+1 in the paper

#? register_forward_hook maybe for adding the new outputs

class FPN(nn.Module):
    def __init__(self, backbone: nn.Module, d=256, batch_size=BATCH_SIZE) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self._d = d
        self.batch_size = batch_size
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.ps = nn.ModuleDict({ #* these are the 3x3 maps added after each merge
            f'p{i+1}': nn.Conv2d(in_channels=self._d, out_channels=self._d, kernel_size=3, padding=1, bias=False)
                for i in range(1, 5) #TODO: check how many layers the backbone has to make it more automatic
            })
    
    def _lateral(self, C_i, x_C, x_P, P_i_plus_1=nn.Identity(), sum_components=True): #* merge operation of C_i with upsampled P_i-1
        x_C = C_i(x_C)
        reduced_channels_C = nn.Conv2d(in_channels=x_C.shape[1], out_channels=self._d, kernel_size=1).to(device) #? if the image's size is divided enough times to get to a odd dimension, reduced_channels_C and upscaled_P may have different dimensions
        reduced_channels_C = reduced_channels_C(x_C)
        if sum_components:
            x_P = P_i_plus_1(x_P)
            upscaled_P = nn.Upsample(scale_factor=2, mode='nearest').to(device)
            upscaled_P = upscaled_P(x_P)
            return torch.add(upscaled_P, reduced_channels_C)
        else:
            return reduced_channels_C
    
    def _batch_element_forward(self, x: torch.Tensor):
        in_layers = False
        # x = self.backbone(x)
        layers = {}
        last = 0
        for name, subm in self.backbone.named_children(): #* remove last classification layers from the backbone
            if not in_layers or name.find('layer') != -1:
                if name.find('layer') != -1: # started 'layer's
                    in_layers = True
                    #* save layers for lateral connections
                    l = int(re.search("\d", name).group())
                    p = [pi for n, pi in self.ps.named_children()][l-1]
                    layers[name] =  {"level": l, "module": subm, "p_module": p, "x": x}
                    if layers[name]['level'] > last: #? condition probably not needed
                        last = layers[name]['level']
                x = subm(x)
            else: # finished 'layer's
                x = x
        
        for name in sorted(layers.keys(), reverse=True):
            l = layers[name]['level']
            m = layers[name]['module']
            p = layers[name]['p_module']
            m_iminus1_x = layers[name]['x']
            
            # these calls use the x computed from the x at the previus iteration, aka the x after it's been processed by a higher level in the pyramid
            if l == last: 
                x = self._lateral(m, m_iminus1_x, x, sum_components=False)
            else:
                x = self._lateral(m, m_iminus1_x, x, p)
            
        return x
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self._batch_element_forward(x)
        else:
            new_x = []
            for x_i in x:
                x_i = self._batch_element_forward(x_i)
                new_x.append(x_i)
            x = tuple(new_x)
        return x

default_backbone = resnet.resnet50(weights='DEFAULT')
default_backbone.to(device)

class RPN(nn.Module):
    def __init__(self, backbone: nn.Module = default_backbone, k=3, n=3, d=256, batch_size=BATCH_SIZE) -> None:
        super().__init__()
        self.fpn = FPN(backbone, d, batch_size)
        self.heads = nn.ModuleDict({
            f'h_{name}': self._create_head(pi, k, n)
                for name, pi in self.fpn.ps.items() #TODO: check how many layers the backbone has to make it more automatic
            })
        
    def _create_head(self, p, k, n):# nn.Conv2d):
        head = nn.ModuleDict({
            'cls': nn.Sequential(
                nn.Conv2d(in_channels=p.out_channels, out_channels=p.out_channels, kernel_size=n, bias=p.bias),
                nn.Conv2d(in_channels=p.out_channels, out_channels=k, kernel_size=1, bias=p.bias),
            ),
            'reg': nn.Sequential(
                nn.Conv2d(in_channels=p.out_channels, out_channels=p.out_channels, kernel_size=n, bias=p.bias),
                nn.Conv2d(in_channels=p.out_channels, out_channels=4*k, kernel_size=1, bias=p.bias),
            )
        })
        return head
    
    def _batch_element_forward(self, x: torch.Tensor):
        x = self.fpn(x)
        xs_cls = []
        xs_reg = []
        for head in self.heads.values():
            x_cls, x_reg = [v(x) for k,v in head.values()] # cls, reg
            xs_cls.append(x_cls)
            xs_reg.append(x_reg)
        return xs_cls, xs_reg
    
    @staticmethod
    def _fake_batch(x):
        return torch.unsqueeze(x, 0)
    
    def forward(self, x):
        xs_cls = []
        xs_reg = []
        if isinstance(x, torch.Tensor):
            x_cls, x_reg = self._batch_element_forward(self._fake_batch(x))
            xs_cls.append(x_cls)
            xs_reg.append(x_reg)
        else:
            x_is_cls = []
            x_is_reg = []
            for x_i in x:
                x_i_cls, x_i_reg = self._batch_element_forward(self._fake_batch(x_i))
                x_is_cls.append(x_i_cls)
                x_is_reg.append(x_i_reg)
            xs_cls.append(x_is_cls)
            xs_reg.append(x_is_reg)
            
            xs_cls = tuple(xs_cls) #* Tuple: distinguish batces from non batches
            xs_reg = tuple(xs_reg)
        return xs_cls, xs_reg

class RPNLoss(nn.Module):
    def __init__(self, n_cls=256, n_reg=2400, reg_norm=10): #TODO: understand "number of anchor locations" for n_reg
        super().__init__()
        self.n_cls = n_cls
        self.n_reg = n_reg
        self.reg_norm = reg_norm
        
        self.logloss = nn.BCELoss()
        self.smoothL1loss = nn.SmoothL1Loss()
        
    def forward(self, pred, target):
        cls_pred, reg_pred = pred
        cls_taget, reg_target = target
        cls_loss = torch.sum(self.logloss(cls_pred, cls_taget)) / self.n_cls
        reg_loss = torch.sum(torch.mul(cls_pred, self.smoothL1loss(reg_pred, reg_target))) * self.reg_norm / self.n_reg
        
        return torch.add(cls_loss, reg_loss)

model = RPN(backbone=resnet50)
# print(model)
print(f"The model has {count_parameters(model):,} trainable parameters")

#! wrong criterion for object classification
criterion = RPNLoss()
criterion = criterion.to(device)

optimizer = optim.SGD(model.parameters(), lr=3e-3) # could be anything, like adam
model = model.to(device)

N_EPOCHS = 25
train_losses, train_accs, valid_losses, valid_accs = model_training(N_EPOCHS, 
                                                                    model, 
                                                                    train_iterator, 
                                                                    val_iterator, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    device,
                                                                    'rpn.pt')

plot_results(N_EPOCHS, train_losses, train_accs, valid_losses, valid_accs)