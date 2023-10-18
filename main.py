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

from helper import calculate_accuracy, model_training, plot_results, count_parameters, BATCH_SIZE, cache_empty, find_free_anns_ids, repair_empty, BROKEN_PATH, get_iou, TanhModule

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
size = 640, 640
augment = False
if augment:
        train_transforms = transforms_v2.Compose([
                                                    transforms_v2.ToImageTensor(),
                                                    transforms_v2.Resize(size, antialias=True),
                                                    # transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                                    transforms_v2.RandomGrayscale(0.2),
                                                    transforms_v2.RandomVerticalFlip(0.4),
                                                    transforms_v2.RandomHorizontalFlip(),
                                                    transforms_v2.RandomPerspective(p=0.3),
                                                    transforms_v2.RandomResizedCrop(size=size, antialias=True),
                                                    transforms_v2.SanitizeBoundingBox(),
        ])
        test_transforms = transforms_v2.Compose([
                                                    transforms_v2.ToImageTensor(),
                                                    transforms_v2.Resize(size, antialias=True),
                                                    # transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                                    transforms_v2.RandomGrayscale(0.2),
                                                    transforms_v2.RandomVerticalFlip(0.4),
                                                    transforms_v2.RandomHorizontalFlip(),
                                                    transforms_v2.RandomPerspective(p=0.3),
                                                    transforms_v2.RandomResizedCrop(size=size, antialias=True),
                                                    transforms_v2.SanitizeBoundingBox(),
        ])
else:
        train_transforms = transforms_v2.Compose([
                                                    transforms_v2.ToImageTensor(),
                                                    transforms_v2.Resize(size, antialias=True),
                                                    # transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                                    transforms_v2.SanitizeBoundingBox(),
        ])
        test_transforms = transforms_v2.Compose([
                                                    transforms_v2.ToImageTensor(),
                                                    transforms_v2.Resize(size, antialias=True),
                                                    # transforms_v2.SanitizeBoundingBox(labels_getter=labels_getter),
                                                    transforms_v2.SanitizeBoundingBox(),
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
        self.tanh = TanhModule()
        for param in self.tanh.parameters():
            param.requires_grad_(False)
        self.heads = nn.ModuleDict({
            f'h_{name}': self._create_head(pi, k, n)
                for name, pi in self.fpn.ps.items() #TODO: check how many layers the backbone has to make it more automatic
            })
        
    def _create_head(self, p, k, n):# nn.Conv2d, k, n):
        r_conv = nn.Conv2d(in_channels=p.out_channels, out_channels=4*k, kernel_size=1, bias=p.bias)
        r = r_conv
        # r = nn.utils.parametrize.register_parametrization(r_conv, "weight", TanhModule())
        head = nn.ModuleDict({
            'cls': nn.Sequential(
                nn.Conv2d(in_channels=p.out_channels, out_channels=p.out_channels, kernel_size=n, bias=p.bias),
                nn.Conv2d(in_channels=p.out_channels, out_channels=k, kernel_size=1, bias=p.bias),
            ),
            'reg': nn.Sequential(
                nn.Conv2d(in_channels=p.out_channels, out_channels=p.out_channels, kernel_size=n, bias=p.bias),
                r,
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
            xs_reg = [self.tanh(x_reg) for x_reg in xs_reg]
            xs_reg = [x_reg.mul(size[0]).div(2) for x_reg in xs_reg]
            # print(f"-- Batch total time: {(time.time() - start):.2f}")
            xs_cls = tuple(xs_cls) #* Make them tuples to distinguish batces from non batches
            xs_reg = tuple(xs_reg)
        return xs_cls, xs_reg

class RPNLoss(nn.Module):
    def __init__(self, reg_norm=10): #TODO: no hardcode n_cls, n_reg in init
        super().__init__()
        # self.n_cls = BATCH_SIZE
        # self.n_reg = 2400 #? "number of anchor locations" but not sure how they get this number
        self.reg_norm = reg_norm
        self.NEGATIVE_INDICATOR = -1
        self.IGNORED_INDICATOR = -2
        self.low_threshold = 0.3
        self.high_threshold = 0.7
        self.sample_size = 256
        
        self.logloss = nn.functional.binary_cross_entropy_with_logits
        self.smoothL1loss = nn.functional.smooth_l1_loss
        
    def forward(self, preds, ds_target):
        print("-- Computing loss...")
        num_imgs = len(ds_target)
        cls_preds, reg_preds = preds
        targets = []
        start_loss = time.time()
        for y_i in ds_target:
            t_i = {'labels': y_i['labels'], 'boxes': y_i['boxes'] }
            targets.append(t_i)
        
        cls_preds, reg_preds = self._concat_boxes(cls_preds, reg_preds)
        
        reg_preds = reg_preds.view(num_imgs, -1, 4)
        labels, matched_gt_boxes = self._assign_targets_to_anchors(reg_preds, targets)
        reg_preds = reg_preds.view(-1, 4)
        
        cls_loss, reg_loss = self._compute_loss(cls_preds, reg_preds, labels, matched_gt_boxes)
        
        print(f"   Loss time: {(time.time() - start_loss):.2f}s")
        return torch.add(cls_loss, reg_loss)

    def _permute_and_flatten(self, head, N, A, C, H, W):
        head = head.view(N, -1, C, H, W)
        head = head.permute(0, 3, 4, 1, 2)
        head = head.reshape(N, -1, C)
        return head

    def _concat_boxes(self, boxes_cls, boxes_reg):
        box_cls_flattened = []
        box_reg_flattened = []
        for box_cls_per_level, box_reg_per_level in zip(boxes_cls, boxes_reg):
            N, AxC, H, W = box_cls_per_level.shape
            Ax4 = box_reg_per_level.shape[1]
            A = Ax4 // 4
            C = AxC // A
            box_cls_per_level = self._permute_and_flatten(box_cls_per_level, N, A, C, H, W)
            box_cls_flattened.append(box_cls_per_level)

            box_reg_per_level = self._permute_and_flatten(box_reg_per_level, N, A, 4, H, W)
            box_reg_flattened.append(box_reg_per_level)
        box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)#.detach()
        box_reg = torch.cat(box_reg_flattened, dim=1).reshape(-1, 4)#.detach()
        return box_cls, box_reg
    
    def _assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            bg_boxes_exist = False #gt_boxes.any(gt_boxes.eq(0))
            if gt_boxes.numel() == 0 or bg_boxes_exist:
                # Background image (negative example) #* can probs leave this as is because the only bg images are the ones i made (so 1 box only)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = get_iou(gt_boxes, anchors_per_image)
                matched_idxs = self._proposal_matcher(match_quality_matrix)
                
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                
                # Negative examples
                bg_indices = matched_idxs == self.NEGATIVE_INDICATOR
                labels_per_image[bg_indices] = 0.0
                # Irrelevant example
                inds_to_discard = matched_idxs == self.IGNORED_INDICATOR
                labels_per_image[inds_to_discard] = -1.0
                
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes
                
    def _proposal_matcher(self, match_quality_matrix):
        matched_vals, matches = match_quality_matrix.max(dim=0)
        all_matches = matches.clone()
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.NEGATIVE_INDICATOR
        matches[between_thresholds] = self.IGNORED_INDICATOR
        
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        # tuple(tensor([0, 1, 1, 2, 2, 3, 3, 4, 5, 5]),
        #           tensor([39796, 32055, 32070, 39190, 40255, 40390, 41455, 45470, 45325, 46390]))
        # Each element in the first tensor is a gt index, and ea3ch element in second tensor is a prediction index
        # Note how gt items 1, 2, 3, and 5 each have two ties
        # So in my case, with 11k ties, it's repeated 11k times
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
        return matches

    def _compute_loss(self, cls_preds, reg_preds, labels, reg_targets):
        cls_preds = cls_preds.flatten()
        
        labels = torch.cat(labels, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)
        
        cls_loss = self.logloss(cls_preds, labels)
        reg_loss = self.smoothL1loss(reg_preds, reg_targets, beta=1/9, reduction="sum") / reg_preds.numel()      

        return cls_loss, reg_loss

model = RPN(backbone=resnet50)
# print(model)
# print(f"The model has {count_parameters(model):,} trainable parameters")

criterion = RPNLoss()
criterion = criterion.to(device)

optimizer = optim.SGD(model.parameters(), lr=3e-3) # could be anything, like adam
model = model.to(device)

N_EPOCHS = 25
train_losses, valid_losses, train_mem, valid_mem = model_training(N_EPOCHS, 
                                                                    model, 
                                                                    train_iterator, 
                                                                    val_iterator, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    device,
                                                                    'rpn',
                                                                    0.005
                                                                    )

# plot_results(N_EPOCHS, train_losses, valid_losses)#, train_accs, valid_accs)
plot_results(N_EPOCHS, train_losses, valid_losses, train_mem, valid_mem)