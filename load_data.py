import os
import random
import re
import torch
import torchvision
import nibabel as nib

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import namedtuple
from torchvision import transforms, utils
from PIL import Image

from os.path import join, isfile
from os import listdir

#constants
TRAIN_DIR = 'img/'
IMG_PREFIX = 'img'
LABEL_DIR = 'label/'
LABEL_PREFIX = 'label'
EXT = '.nii.gz'
SPLEEN_VAL = 1


def process_image(img_file, normalize=True):

    if not img_file:
        return None

    img_3d = nib.load(img_file)
    img = img_3d.get_data()

    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 255.0

    img = np.transpose(img, (2,0,1))

    return img

class Img:
    def __init__(self, img, label , axis, idx):
        self.img = img
        self.label = label
        self.axis = axis
        self.idx = idx
        self.complete = False


class SpleenDataset(Dataset):
    def __init__(self, root_dir, img_range=(1,2), axis=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_range = img_range
        self.cur_sample = Img(None, None, 0, 0)
        self.cur_sample.complete = True
        self.img_num = img_range[0]
        self.len = 0

        #check if there is a labels
        if self.root_dir[-1] != '/':
            self.root_dir += '/'

        self.is_labeled = os.path.isdir(self.root_dir + LABEL_DIR)

        #compute the total number of frames
        for img_num in range(img_range[0], img_range[1]+1):
            img_file = os.path.join(self.root_dir, TRAIN_DIR, IMG_PREFIX + str(img_num).zfill(4) + EXT)
            print(img_file)
            self.len += process_image(img_file, False).shape[0]

        print(self.len)
        self.img_num -= 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if all slices of the current sample are exhausted (continue)
        if self.cur_sample.complete:
            self.img_num += 1
            img_file = os.path.join(self.root_dir, TRAIN_DIR, IMG_PREFIX + str(self.img_num).zfill(4) + EXT)
            label_file = os.path.join(self.root_dir, LABEL_DIR, LABEL_PREFIX + str(self.img_num).zfill(4) + EXT) if self.is_labeled else None

            self.cur_sample = Img(process_image(img_file), process_image(label_file, False), 0, 0) #img, label, axis, idx
            print(self.cur_sample.img.shape)

        #self.cur_sample should be ready
        prev_img_slice = self.cur_sample.img[max(self.cur_sample.idx - 1, 0), :, :]
        img_slice = self.cur_sample.img[self.cur_sample.idx, : , :]
        next_img_slice = self.cur_sample.img[min(self.cur_sample.idx + 1, self.cur_sample.img.shape[0] - 1), :, :]

        prev_img_slice = prev_img_slice.astype('float32')
        img_slice = img_slice.astype('float32')
        next_img_slice = next_img_slice.astype('float32')

        img_label = self.cur_sample.label[self.cur_sample.idx, :, :] if self.is_labeled else np.array([])
        self.cur_sample.idx += 1

        self.cur_sample.complete = self.cur_sample.idx == self.cur_sample.img.shape[0]

        #if we have an image label filter for the spleen
        if img_label.size != 0:
            img_label = img_label == SPLEEN_VAL

        img_label = img_label.astype('float32')

        #im = Image.fromarray(np.uint8(img_slice))
        #im.save('./gen/gen_' + str(idx).zfill(4) + ".png")

        # convert to tensors
        imgcs = torch.from_numpy(img_slice).unsqueeze(0)
        imgns = torch.from_numpy(next_img_slice).unsqueeze(0)
        imgps = torch.from_numpy(prev_img_slice).unsqueeze(0)
        mask = torch.from_numpy(img_label).unsqueeze(0)

        return imgps, imgcs, imgps, mask

