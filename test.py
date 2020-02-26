# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp

# general libs
import cv2
from PIL import Image
import numpy as np
import datetime
import math
import time
import os
import sys
import glob
import argparse

### My libs
from utils.helpers import *
from models.OPN import OPN
from models.TCN import TCN
from models.dataset import dataset

parser = argparse.ArgumentParser(description="CPNet")
parser.add_argument("-b", type=int, default=1)
parser.add_argument("-e", type=int, default=0)
parser.add_argument("-n", type=str, default='youtube-vos') 
parser.add_argument("-m", type=str, default='fixed') 
args = parser.parse_args()

BATCH_SIZE = args.b
RESUME = args.e
DATA_NAME = args.n
MASK_TYPE = args.m

w,h = 424, 240
default_fps = 6
# every 5 frame as memory frames
MEM_EVERY = 5 


# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def get_clear_state_dict(old_state_dict):
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k,v in old_state_dict.items():
    name = k 
    if k.startswith('module.'):
      name = k[7:]
    new_state_dict[name] = v
  return new_state_dict


def main_worker(gpu, ngpus_per_node, args):
  if ngpus_per_node > 0:
    torch.cuda.set_device(int(gpu))
  # set random seed 
  seed = 2020
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  #################### Load Model
  model = set_device(OPN())
  data = torch.load('weights/OPN.pth', map_location = lambda storage, loc: set_device(storage))
  model.load_state_dict(get_clear_state_dict(data), strict=False)
  model.eval() 

  pp_model = set_device(TCN()) 
  data = torch.load('weights/TCN.pth', map_location = lambda storage, loc: set_device(storage))
  pp_model.load_state_dict(get_clear_state_dict(data), strict=False)
  pp_model.eval() 


  Pset = dataset(DATA_NAME, MASK_TYPE)
  step = math.ceil(len(Pset) / ngpus_per_node)
  Pset = torch.utils.data.Subset(Pset, range(gpu*step, min(gpu*step+step, len(Pset))))
  Trainloader = torch.utils.data.DataLoader(Pset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


  save_path = 'results/{}_{}'.format(DATA_NAME, MASK_TYPE)
  for vi, V in enumerate(Trainloader):
    frames, masks, GTs, dists, info = V # b,3,t,h,w / b,1,t,h,w
    frames, masks, dists, GTs = set_device([frames, masks, dists, GTs])
    seq_name = info['name'][0]
    T = frames.size()[2]
    print('[{}] {}/{}: {} for {} frames ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
      vi, len(Trainloader), seq_name, frames.size()[2]))

    with torch.no_grad():
      # valids area
      valids = (1-masks).float()
      ################### Inference
      comps = torch.zeros_like(frames)
      ppeds = torch.zeros_like(frames)

      # memory encoding 
      midx = list( range(0, T, MEM_EVERY))
      mkey, mval, mhol = model(frames[:,:,midx], valids[:,:,midx], dists[:,:,midx])

      # inpainting
      for f in range(T):
        # memory selection
        if f in midx:
          ridx = [i for i in range(len(midx)) if i != int(f/MEM_EVERY)]
        else:
          ridx = list(range(len(midx)))

        fkey, fval, fhol = mkey[:,:,ridx], mval[:,:,ridx], mhol[:,:,ridx]
        for r in range(999): 
          if r == 0:
            comp = frames[:,:,f]
            dist = dists[:,:,f]
          with torch.no_grad(): 
            comp, dist = model(fkey, fval, fhol, comp, valids[:,:,f], dist)
          
          # update
          comp, dist = comp.detach(), dist.detach()
          if torch.sum(dist).item() == 0:
            break
            
        comps[:,:,f] = comp

      # post-processing...
      ppeds[:,:,0] = comps[:,:,0]
      hidden = None
      for f in range(T):
        pped, hidden = pp_model(ppeds[:,:,f-1], masks[:,:,f-1], comps[:,:,f], masks[:,:,f], hidden)
        ppeds[:,:,f] = pped

    os.makedirs(os.path.join(save_path, seq_name), exist_ok=True)
    comp_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'comp.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    pred_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'pred.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    mask_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'mask.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    orig_writer = cv2.VideoWriter(os.path.join(save_path, seq_name, 'orig.avi'),
      cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
    for f in range(T):
      est = (ppeds[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8)
      true = (GTs[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8) # h,w,3
      mask = np.expand_dims((dists[0,0,f].detach().cpu().numpy() > 0).astype(np.uint8), axis=2) # h,w,1
      comp_writer.write(cv2.cvtColor(true*(1-mask)+est*mask, cv2.COLOR_BGR2RGB))
      pred_writer.write(cv2.cvtColor(est, cv2.COLOR_BGR2RGB))
      mask_writer.write(cv2.cvtColor(true*(1-mask)+255*mask, cv2.COLOR_BGR2RGB))
      orig_writer.write(cv2.cvtColor(true, cv2.COLOR_BGR2RGB))
    comp_writer.release()
    pred_writer.release()
    mask_writer.release()
    orig_writer.release()
  print('Finish in {}'.format(save_path))




if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  if ngpus_per_node > 0:
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    main_worker(0, 1, args)
