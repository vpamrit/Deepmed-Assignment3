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
    def __init__(self, root_dir, img_range=(1,1), axis=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_range = img_range
        self.cur_sample = Img(None, None, 0, 0)
        self.cur_sample.complete = True

        #check if there is a labels
        if self.root_dir[-1] != '/':
            self.root_dir += '/'

        self.is_labeled = os.path.isdir(self.root_dir + LABEL_DIR)

    #this is actually incorrect
    def __len__(self):
        return (self.img_range[1] - self.img_range[0] + 1)*147

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_num = 1  #idx + self.img_range[0]


        if self.cur_sample.complete:
            img_file = os.path.join(self.root_dir, TRAIN_DIR, IMG_PREFIX + str(img_num).zfill(4) + EXT)
            label_file = os.path.join(self.root_dir, LABEL_DIR, LABEL_PREFIX + str(img_num).zfill(4) + EXT) if self.is_labeled else None

            self.cur_sample = Img(process_image(img_file), process_image(label_file, False), 0, 0)

        #self.cur_sample should be ready
        img_slice = self.cur_sample.img[self.cur_sample.idx, : , :]
        img_slice = img_slice.astype('float32')

        img_label = self.cur_sample.label[self.cur_sample.idx, :, :] if self.is_labeled else np.array([])
        self.cur_sample.idx += 1

        #if we have an image label filter for the spleen
        if img_label.size != 0:
            img_label = img_label == SPLEEN_VAL

        img_label = img_label.astype('float32')

        im = Image.fromarray(np.uint8(img_label))
        im.save('./gen/gen_' + str(idx).zfill(4) + ".png")

        # numpy is W x H x C
        # torch is C x H x W
        #image = torch.from_numpy(image.transpose((2, 0, 1)))
        #label = torch.from_numpy(self.labels[idx, 1:3].astype('float').reshape(-1,2).squeeze())

        return img_slice, img_label

