import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import re

MODELS_PATH = 'models/'
BATCH_SIZE = 256

# TODO: check if it's correct for this project
def calculate_accuracy(y_pred, y):
  '''
  Compute accuracy from ground-truth and predicted labels.
  
  Input
  ------
  y_pred: torch.Tensor [BATCH_SIZE, N_LABELS]
  y: torch.Tensor [BATCH_SIZE]

  Output
  ------
  accuracy: np.float
    Accuracy
  '''
  
  # Apply softmax (to convert values into probabilities)
  y_prob = F.softmax(y_pred, dim = -1)
  # Consider class with higher probability
  y_pred = y_pred.argmax(dim=1, keepdim = True)
  # Chech if prediction is correct
  correct = y_pred.eq(y.view_as(y_pred)).sum()
  # Compute accuracy (percentage)
  accuracy = correct.float()/y.shape[0]

  return accuracy

def train(model, iterator, optimizer, criterion, device, batches_percentage=0.7):
  epoch_loss = 0
  # epoch_acc = 0
  
  # Apply train mode
  model.train()
  mem = []
  iters = 0
  max_batches = int(len(iterator)*batches_percentage)
  for (x,y) in iterator:
    # if iters % 100 == 0: 
    #   if iters!= 0: 
    #     for i in range(5): print(UP, end=CLEAR)
    print(f"BATCH {iters+1}/{max_batches}")
    print("Allocated before gc: ", torch.cuda.memory_allocated()*1e-9)
    gc.collect()
    print("Allocated: ", torch.cuda.memory_allocated()*1e-9)
    print("Reserved: ", torch.cuda.memory_reserved()*1e-9)
    torch.cuda.empty_cache()
    mem.append(torch.cuda.memory_allocated()*1e-9)
    print("Reserved after empty cache: ", torch.cuda.memory_reserved()*1e-9)
    x = list(x)
    shapes = []
    for i, x_i in enumerate(x):
      x[i] = x_i.to(device).float()
      shapes.append((x_i.shape[1], x_i.shape[2]))
    x = tuple(x)
    
    if type(y) != list and type(y) != tuple:
      y = y.to(device)
    elif type(y) == tuple:
      for i, y_i in enumerate(y):
        for k in y_i.keys():
          if isinstance(y_i[k], torch.Tensor): # boxes, masks, labels
            y[i][k] = y_i[k].to(device)
        y[i]['img_size'] = shapes[i][-2:]
    else:
      y = [y_i.to(device) for y_i in y]
    
    # Set gradients to zero
    optimizer.zero_grad()
    
    # Make Predictions
    y_pred = model(x)
    mem.append(torch.cuda.memory_allocated()*1e-9)
    torch.cuda.empty_cache()

    # Compute loss
    loss = criterion(y_pred, y)
    torch.cuda.empty_cache()
    
    # Compute accuracy
    # acc = calculate_accuracy(y_pred, y)

    # Backprop
    bw_start = time.time()
    loss.backward()
    torch.cuda.empty_cache()
    print(f"-- Backward time: {(time.time() - bw_start):.2f}s")
    
    # Apply optimizer
    optimizer.step()

    # Extract data from loss and accuracy
    epoch_loss += loss.item()
    # epoch_acc += acc.item()
    del loss
    
    iters += 1
    if iters >= max_batches: break
  return epoch_loss/iters, 0, mem#epoch_acc/len(iterator)

def evaluate(model, iterator, criterion, device, batches_percentage=0.4):
  epoch_loss = 0
  # epoch_acc = 0

  # Evaluation mode
  model.eval()

  # Do not compute gradients
  with torch.no_grad():
    iters=0
    mem_val=[]
    max_batches = int(len(iterator)*batches_percentage)
    for (x,y) in iterator:
      print(f"BATCH {iters+1}/{max_batches}")
      print("Allocated before gc: ", torch.cuda.memory_allocated()*1e-9)
      gc.collect()
      print("Allocated: ", torch.cuda.memory_allocated()*1e-9)
      print("Reserved: ", torch.cuda.memory_reserved()*1e-9)
      torch.cuda.empty_cache()
      mem_val.append(torch.cuda.memory_allocated()*1e-9)
      print("Reserved after empty cache: ", torch.cuda.memory_reserved()*1e-9)
      x = list(x)
      shapes = []
      for i, x_i in enumerate(x):
        x[i] = x_i.to(device).float()
        shapes.append((x_i.shape[1], x_i.shape[2]))
      x = tuple(x)
      
      if type(y) != list and type(y) != tuple:
        y = y.to(device)
      elif type(y) == tuple:
        for i, y_i in enumerate(y):
          for k in y_i.keys():
            if isinstance(y_i[k], torch.Tensor): # boxes, masks, labels
              y[i][k] = y_i[k].to(device)
          y[i]['img_size'] = shapes[i][-2:]
      else:
        y = [y_i.to(device) for y_i in y]
      
      # Make Predictions
      y_pred = model(x)
      mem_val.append(torch.cuda.memory_allocated()*1e-9)
      torch.cuda.empty_cache()

      # Compute loss
      loss = criterion(y_pred, y)
      torch.cuda.empty_cache()
      
      # Compute accuracy
      # acc = calculate_accuracy(y_pred, y)

      # Extract data from loss and accuracy
      epoch_loss += loss.item()
      # epoch_acc += acc.item()
      del loss
      
      iters += 1
      if iters >= max_batches: break
  return epoch_loss/iters, 0, mem_val#epoch_acc/len(iterator)

def model_training(n_epochs, model, train_iterator, val_iterator, optimizer, criterion, device, model_name='models/best_model', batches_percentage=0.7, load=None):
  # models saved in subfolder
  if not model_name.startswith((f'{MODELS_PATH}', f'./{MODELS_PATH}')):
    model_name = f'{MODELS_PATH}{model_name}'
  # model to more suitable hw
  model.to(device)
  # Initialize validation loss
  best_valid_loss = float('inf')

  # Save output losses, accs
  
  train_losses = []
  # train_accs = []
  valid_losses = []
  # valid_accs = []
  train_mem = []
  valid_mem = []

  if load is not None:
    model.load_state_dict(torch.load(load))
    n = re.findall(r'\d+', load)
    n = list(map(int, n))[0]
    range_epochs = range(n,n_epochs)
  else:
    range_epochs = range(n_epochs)
  
  # Loop over epochs
  for epoch in range_epochs:
    torch.save(model.state_dict(), f'{MODELS_PATH}/epoch_{epoch}_before_training.pt')
    start_time = time.time()
    
    print(f"\nEpoch: {epoch+1}/{n_epochs}")
    print("---------------------------------")
    # Train
    print("   Training")
    print("--------------")
    train_loss, train_acc, tm = train(model, train_iterator, optimizer, criterion, device, batches_percentage)
    torch.save(model.state_dict(), f'{MODELS_PATH}/epoch_{epoch}_between_training.pt')
    # Validation
    print("--------------")
    print("  Validation")
    print("--------------")
    valid_loss, valid_acc, vm = evaluate(model, val_iterator, criterion, device, batches_percentage/2)
    # Save best model
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      # Save model
      torch.save(model.state_dict(), f'{model_name}_ep{epoch}.pt')
    end_time = time.time()
    
    print(f"Time: {end_time-start_time:.2f} s")
    print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_acc * 100:.2f}%")
    print(f"Val -- Loss: {valid_loss:.3f}, Acc: {valid_acc * 100:.2f}%")

    # Save
    train_losses.append(train_loss)
    # train_accs.append(train_acc)
    train_mem.append(tm)
    valid_losses.append(valid_loss)
    # valid_accs.append(valid_acc)
    valid_mem.append(vm)

    torch.save(model.state_dict(), f'{MODELS_PATH}/epoch_{epoch}_after_training.pt')
  return train_losses, valid_losses, train_mem, valid_mem#train_accs, valid_losses, valid_accs

def plot_results(n_epochs, train_losses, train_accs, valid_losses, valid_accs):
  N_EPOCHS = n_epochs
  # Plot results
  plt.figure(figsize=(20, 6))
  _ = plt.subplot(1,2,1)
  plt.plot(np.arange(N_EPOCHS)+1, train_losses, linewidth=3)
  plt.plot(np.arange(N_EPOCHS)+1, valid_losses, linewidth=3)
  _ = plt.legend(['Train', 'Validation'])
  _ = plt.grid(True), plt.xlabel('Epoch'), plt.ylabel('Loss')

  _ = plt.subplot(1,2,2)
  plt.plot(np.arange(N_EPOCHS)+1, train_accs, linewidth=3)
  plt.plot(np.arange(N_EPOCHS)+1, valid_accs, linewidth=3)
  _ = plt.legend(['Train', 'Validation'])
  _ = plt.grid(True), plt.xlabel('Epoch'), plt.ylabel('Accuracy')

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad) # Count only parameters that are backpropagated

UP = '\033[1A'
CLEAR = '\x1b[2K'
BROKEN_PATH = 'broken/'

def cache_empty(dataset, ds_name):
    broken = []
    print('Progress:\n')
    for i in range(len(dataset)):
        sample = dataset[i]
        img, target = sample
        if(i%500 == 0):
            print(UP,end=CLEAR)
            print(f'\r{i+1}/{len(dataset)}')
        if not ('labels' in target.keys() and 'segmentation' in target.keys()):
            broken.append({'idx': i, 'img_id': target['image_id'], 'keys': target.keys()})
            print(UP,end=CLEAR)
            print(f'\r{i}: {target["image_id"]}, {target.keys()}\n')
    with open(f'{BROKEN_PATH}{ds_name}.json', 'w') as f:
        json.dump([i['idx'] for i in broken], f)
    with open(f'{BROKEN_PATH}{ds_name}.json', 'w') as f:
        json.dump([i['img_id'] for i in broken], f)

def repair_empty(dataset, ds_name, use_cache, ann_ids):
    print(f'Repairing {ds_name}:')
    with open(f'data/annotations/instances_{ds_name}2017.json', 'r') as annsFile:
        contents = json.load(annsFile)
    anns = contents['annotations']
    if use_cache:
        with open(f'{BROKEN_PATH}{ds_name}.json', 'r') as f:
            checklist = json.load(f)
    else:
        checklist = range(len(dataset))
    print('Progress:\n')
    for i in checklist:
      sample = dataset[i]
      img, target = sample
      img_id = target['image_id']
      if(i%(5 if not use_cache and ds_name == 'val' else 500) == 0):
        print(UP,end=CLEAR)
        print(f'\r{i+1}/{len(dataset)}')
      if use_cache or not ('labels' in target.keys() and 'segmentation' in target.keys()):
        c, h, w = img.size()
        bbox =  [0, 0, w, h]
        seg = [0, 0, 0, h, w, h, w, 0]
        seg = [[float(n) for n in seg]]
        area = float(w*h)
        target = {'segmentation': seg, 'area': area, 'iscrowd': 0, 'image_id': img_id, 'bbox': bbox, 'category_id': 0, 'id': ann_ids.pop(0)} #! cat 0 is background, could be a problem
        anns.append(target)
        # print(i, target['image_id'])
    contents['annotations'] = anns
    with open(f'data/annotations/instances_{ds_name}2017.json', 'w') as annsFile:
      json.dump(contents, annsFile)
        
def find_free_anns_ids(anns_sorted, free_ann_ids, missing):
    last = 1
    finished = False
    for ann_id in anns_sorted:
      if ann_id != last+1 and not finished:
        for i in range(last, ann_id):
          if not finished:
            free_ann_ids.append(i)
            if len(free_ann_ids) >= missing:
              finished = True
              break
      elif finished:
        break
      last = ann_id
      
# def _clip_negative_corner(a):
#   return [x if x >= 0 else 0 for x in a]
# def _order_corners(c1, c2):
#   return (c1, c2) if c1 <= c2 else (c2, c1)
# def _compare_corners(c1,c2, d1, d2):
#   w1 = min(c1,d1)
#   w2 = c2 if w1 == c1 else d2
#   z1 = max(c1,d1)
#   z2 = d2 if z1 == d1 else c2
#   return w1, w2, z1, z2
# def _segment_intersection(w2, z1, z2): #w1 will never be > z1 because it's computed as the min
#   return min(z2, w2) - z1 if z1 < w2 else 0 

# def get_iou(a, b, max_size=0): #todo (maybe): sanitize for image size
#   a = _clip_negative_corner(a)
#   b = _clip_negative_corner(b)
#   ax1, ay1, ax2, ay2 = a
#   bx1, by1, bx2, by2 = b
#   ax1, ax2 = _order_corners(ax1, ax2)
#   ay1, ay2 = _order_corners(ay1, ay2)
#   bx1, bx2 = _order_corners(bx1, bx2)
#   by1, by2 = _order_corners(by1, by2)

#   # w refers to the box with the top-left corner closest to the origin in that axis
#   wx1, wx2, zx1, zx2 = _compare_corners(ax1, ax2, bx1, bx2)
#   wy1, wy2, zy1, zy2 = _compare_corners(ay1, ay2, by1, by2)
  
#   hor_side = _segment_intersection(wx2, zx1, zx2)
#   ver_side = _segment_intersection(wy2, zy1, zy2)
  
#   intersection = hor_side * ver_side
#   area_a = (ax2-ax1) * (ay2-ay1)
#   area_b = (bx2-bx1) * (by2-by1)
#   union = area_a + area_b - intersection
  
#   return intersection / union if union != 0 else 0

def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
def get_area(b):
    b = _upcast(b)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
def get_iou(boxes1, boxes2) :
    area1 = get_area(boxes1)
    area2 = get_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - intersection
    
    return intersection/union