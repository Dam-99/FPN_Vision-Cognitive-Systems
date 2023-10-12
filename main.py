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
import itertools as it
# For results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# COCO
from pycocotools.coco import COCO

from helper import calculate_accuracy, model_training, plot_results, count_parameters, BATCH_SIZE, cache_empty, find_free_anns_ids, repair_empty, BROKEN_PATH, get_iou

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
        
    def _create_head(self, p, k, n):# nn.Conv2d, k, n):
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
    
    def _batch_element_fpn_run(self, x: torch.Tensor):
        x = self.fpn(x)
        return x
    
    def _rpn_forward(self, x: torch.Tensor):
        xs_cls = []
        xs_reg = []
        for head in self.heads.values():
            x_cls, x_reg = [v(x) for v in head.values()]
            xs_cls.append(x_cls)
            xs_reg.append(x_reg)
        return xs_cls, xs_reg #* understand how to return the values here so that they can be used later (also for the forward result) -- a tuple should be ok
    
    @staticmethod
    def _fake_batch(x):
        return torch.unsqueeze(x, 0)
    
    def forward(self, x: torch.Tensor | tuple[torch.Tensor]):
        xs_cls = []
        xs_reg = []
        # After going through fpn, everything should be of the same size so i can batch it and optimize stuff a bit
        if isinstance(x, torch.Tensor):
            x = self._batch_element_fpn_run(x)#self._fake_batch(x))
            x_cls, x_reg = self._rpn_forward(x)
            xs_cls.append(x_cls)
            xs_reg.append(x_reg)
        else:
            bi = 0
            # print("-- Batch images' times:")
            # start = time.time()
            # TODO: find a way to not hardcode it
            batch_x_shape = len(x), self.fpn._d, int(x[0].shape[1]/4), int(x[0].shape[2]/4) # batch_size, channels, h, w
            batch_x = torch.empty(size=batch_x_shape).to(device)
            for i, x_i in enumerate(x):
                start_i = time.time()
                batch_x[i] = self._batch_element_fpn_run(self._fake_batch(x_i)).squeeze(dim=0)
                print(f"     {bi+1}/{BATCH_SIZE}: {(time.time() - start_i):.2f}s")
                bi+=1
            xs_cls, xs_reg = self._rpn_forward(batch_x)
            # print(f"-- Batch total time: {(time.time() - start):.2f}")
            xs_cls = tuple(xs_cls) #* Make them tuples to distinguish batces from non batches
            xs_reg = tuple(xs_reg)
        return xs_cls, xs_reg

class RPNLoss(nn.Module):
    def __init__(self, reg_norm=10): #TODO: no hardcode n_cls, n_reg in init
        super().__init__()
        self.n_cls = BATCH_SIZE
        self.n_reg = 2400 #? "number of anchor locations" but not sure how they get this number
        self.reg_norm = reg_norm
        
        self.logloss = nn.functional.binary_cross_entropy
        self.smoothL1loss = nn.functional.smooth_l1_loss
        
    def forward(self, pred, target):
        print("-- Computing loss...")
        cls_pred, reg_pred = pred
        cls_target = [] # these needs to be tensors
        reg_target = []
        start_loss = time.time()
        for y_i in target:
            t_i = torch.any(torch.ne(y_i['labels'], 0)).to(device), y_i['boxes'] 
            cls_target.append(t_i[0])
            reg_target.append(t_i[1])
        
        cls_loss, reg_loss = self._compute_losses_tensors(cls_pred, reg_pred, reg_target)
        cls_loss = torch.div(torch.sum(cls_loss).item(), self.n_cls)
        reg_loss = torch.div(torch.sum(reg_loss).item(), self.n_reg)
        print(f"   Loss time: {(time.time() - start_loss):.2f}s")
        return torch.add(cls_loss, reg_loss)
    
    def _compute_losses_tensors(self, anchors, anchors_boxes, reg_target):
        b_size = len(reg_target)
        n_heads = len(anchors)
        img_n_anchor, img_w, img_h = anchors[0][0].shape
        
        cls_pred = torch.stack([head.clone().detach() for head in anchors]).permute(1,0,3,4,2).contiguous()
        reg_pred = torch.stack([head.clone().detach() for head in anchors_boxes]).permute(1,0,3,4,2).view(b_size, n_heads, img_w, img_h, img_n_anchor, 4).contiguous()
        labels = torch.full(size=cls_pred.shape, fill_value=-1, dtype=torch.float32).contiguous().to(device)
        labels_gtb = torch.zeros(size=cls_pred.shape, dtype=torch.int).contiguous()
        anchor_count_wrong = torch.zeros(size=cls_pred.shape, dtype=torch.int).contiguous()
        cls_loss = torch.zeros(size=torch.Size([b_size])).contiguous() # tensors containing the loss for each image
        reg_loss = torch.zeros(size=torch.Size([b_size])).contiguous()
        
        gtb_max_anchor = []
        for t in reg_target:
            # no_bg_target_boxes.append(t['boxes'][t['labels'].ne(0)]) #? class 0 already removed in forward
            gtb_max_anchor.append(torch.zeros(size=torch.Size([len(t),5]), dtype=torch.int))
            
        gtb_count = 0
        heads, widths, heights, anchs = range(n_heads), range(img_w), range(img_h), range(img_n_anchor)
        for i, t in enumerate(reg_target): # i represents the batch size/element, t is a tensor with all the boxes's coordinates
            tbs = t
            gtb_count += len(tbs) # should be just the number of boxes, not also ,4
            for head, w, h, a in it.product(heads, widths, heights, anchs):
                skip = False
                for j, gtb_tensor in enumerate(tbs):
                    if skip: break
                    anchor = reg_pred[i][head][w][h][a].tolist()
                    gtb = gtb_tensor.tolist()
                    iou = get_iou(anchor, gtb)
                    if iou > gtb_max_anchor[i][j][4]:
                        gtb_max_anchor[i][j] = torch.tensor([head, w, h, a, iou])
                    if iou >= 0.7:
                        labels[i][head][w][h][a] = 1
                        labels_gtb[i][head][w][h][a] = j
                        skip = True
                        continue
                    if iou < 0.3:
                        anchor_count_wrong[i][head][w][h][a] += 1
            # append pred for each image
            # cls_pred += pred[i][labels.ne(-1)]
            # reg_pred += anchors_boxes[i][labels.ne(-1).ne(0)] # this should be equivalent to multiplying it with the label
        labels[anchor_count_wrong.eq(gtb_count)] = 0
        # set anchors' labels to 1 for max gtb
        for i in range(b_size):
            max_gtb_i = gtb_max_anchor[i]
            for coords in max_gtb_i.tolist():
                head,w,h,a,_ = [int(c) for c in coords]
                labels[i][head][w][h][a] = 1
        # filter anchors according to label
        cls_pred = cls_pred[labels.ne(-1)]#.reshape(cls_pred_shape) #TOFIX: shapes 
        reg_pred = reg_pred[labels.ne(-1)]#.reshape(reg_pred_shape)
        labels = labels[labels.ne(-1)] # probably needed for keeping the same shape
        # compute cls_loss
        cls_loss = self.logloss(cls_pred, labels)
        # compute reg_loss using the correct label
        reg_loss = [self.smoothL1loss(reg_pred[i], reg_target[i]) for i in range(b_size)]
        reg_loss = torch.tensor([torch.sum(rli) for rli in reg_loss])
        
        return cls_loss, reg_loss

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