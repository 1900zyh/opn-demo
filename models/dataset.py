import matplotlib
import io
import os
import json
import os.path as osp
import numpy as np
from PIL import Image

import collections
import torch
import torchvision
from torch.utils import data

import glob
import time
import cv2
import random
import csv
import tqdm

import matplotlib.pyplot as plt
import zipfile

class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, 'r')
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im

class dataset(data.Dataset):
  def __init__(self, data_name, mask_type, size=(240,424)):
    with open(os.path.join('../flist', data_name, 'test.json'), 'r') as f:
      self.video_dict = json.load(f)
    self.videos = list(self.video_dict.keys())
    with open(os.path.join('../flist', data_name, 'mask.json'), 'r') as f:
      self.mask_dict = json.load(f)
    self.masks = list(self.mask_dict.keys())
    self.size = size
    self.mask_type = mask_type
    self.data_name = data_name

  def __len__(self):
    return len(self.videos)

  def __getitem__(self, index):
    info = {}
    video = self.videos[index]
    info['name'] = video
    frame_names = self.video_dict[video]
    N = len(frame_names)
    idxs = range(N)
    H, W = self.size
    
    # make sure dividable by 8
    d = 8
    if H % d > 0:
      if H % d > d/2:
        new_H = H + d - H % d
      else:
        new_H = H - H % d
    else:
      new_H = H

    if W % d > 0:
      if W % d > d/2:
        new_W = W + d - W % d
      else:
        new_W = W - W % d
    else:
      new_W = W
    H, W = new_H, new_W

    N_frames = np.empty((N, H, W, 3), dtype=np.float32)
    N_masks = np.empty((N, H, W, 1), dtype=np.float32)
    N_dists = np.empty((N, H, W, 1), dtype=np.float32)
    for i, f in enumerate(frame_names):
      img = ZipReader.imread('../datazip/{}/JPEGImages/{}.zip'.format(self.data_name, video), f)
      raw_frame = np.array(img.convert('RGB'))/255.
      N_frames[i] = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

      raw_mask = self._get_masks(index, video, i)
      raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
      N_dists[i,:,:,0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5).astype(np.float32)
      N_masks[i,:,:,0] = raw_mask.astype(np.float32)

    Fs = torch.from_numpy(np.transpose(N_frames, (3, 0, 1, 2)).copy()).float()
    Hs = torch.from_numpy(np.transpose(N_masks, (3, 0, 1, 2)).copy()).float()
    dts = torch.from_numpy(np.transpose(N_dists, (3, 0, 1, 2)).copy()).float()
  
    GTs = Fs
    Fs = (1-Hs)*GTs + Hs*torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1,1)

    return Fs, Hs, GTs, dts, info


  def _get_masks(self, index, video, i):
    h, w = self.size
    if self.mask_type == 'fixed':
      m = np.zeros(self.size, np.uint8)
      m[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8] = 1
      return m
    elif self.mask_type == 'random_obj':
      m = ZipReader.imread('../datazip/random_masks/{}.zip'.format(self.data_name),\
        '{}.png'.format(video)).resize((w, h))
      m = np.array(m)
      m = np.array(m>0).astype(np.uint8)
      return m
    elif self.mask_type == 'object':
      m_name = self.mask_dict[video][i]
      m = ZipReader.imread('../datazip/{}/Annotations/{}.zip'.format(self.data_name, video), m_name).convert('L')
      m = np.array(m)
      m = np.array(m>0).astype(np.uint8)
      m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=4)
      return m
    else:
      raise NotImplementedError(f"Mask type {self.mask_type} not exists")
