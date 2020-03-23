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
SAVE_IMAGES = False
TRAIN_DIR = 'img/'
IMG_PREFIX = 'img'
LABEL_DIR = 'label/'
LABEL_PREFIX = 'label'
EXT = '.nii.gz'
SPLEEN_VAL = 1

def process_image(img_file, padding=0, normalize=True):

    if not img_file:
        return None

    img_3d = nib.load(img_file)
    img = img_3d.get_data()

    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 255.0


    img = np.transpose(img, (2,0,1))

    if padding != 0:
        npad = ((0, 0), (padding, padding), (padding, padding))
        img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)

    return img

class Img:
    def __init__(self, img_name, img, label , idx, slice_num):
        self.img_name = img_name
        self.img = img
        self.label = label
        self.idx = idx
        self.slice_num = slice_num
        self.complete = False


class SpleenDataset(Dataset):
    def __init__(self, root_dir, img_range=(0,0), slice_size = 240, slice_stride = 80, num_slices = 5, transform=None, classes=[1,2,3,4,5,6,7,8,9,10,11,12,13]):
        self.root_dir = root_dir
        self.transform = transform
        self.img_range = img_range
        self.cur_sample = Img(None, None, None, 0, 0)
        self.cur_sample.complete = True
        self.img_num = img_range[0]
        self.first_img = img_range[0]
        self.last_img = img_range[1]
        self.len = 0
        self.slice_size = int(slice_size)
        self.slice_stride = int(slice_stride)
        self.num_slices = int(num_slices)
        self.total_slices = num_slices * num_slices
        self.classes = classes
        self.samples = []


        #compute the padding here
        self.padding = int (( (num_slices - 1) * slice_stride + slice_size - 512) / 2)
        self.window_size = slice_size - self.padding * 2

        #check if there is a labels
        if self.root_dir[-1] != '/':
            self.root_dir += '/'

        self.is_labeled = os.path.isdir(self.root_dir + LABEL_DIR)

        self.files = [re.findall('[0-9]{4}', filename)[0] for filename in os.listdir(self.root_dir + TRAIN_DIR)]
        self.files = sorted(self.files, key = lambda f : int(f))

        #compute the total number of frames
        for img_num in range(img_range[0], img_range[1]+1):
            img_file = os.path.join(self.root_dir, TRAIN_DIR, IMG_PREFIX + self.files[img_num] + EXT)
            label_file = os.path.join(self.root_dir, LABEL_DIR, LABEL_PREFIX + self.files[self.img_num] + EXT) if self.is_labeled else None

            sample = (self.files[self.img_num], process_image(img_file, self.padding, True), process_image(label_file, self.padding, False)) #img, label, axis, idx

            # create a map of idx to sample | idx to slice_num

            print(img_file)
            self.samples.append(sample)

        for sample in self.samples:
            self.len += self.samples[0].shape[0]

        print("Dataset details\n  Images: {}, 2D Slices: {}, Subslices {}, Padding-Margin: {}".format(self.last_img - self.first_img + 1, self.len, self.total_slices, self.padding))
        self.img_num -= 1

    #return start of next slice for the current sample
    def get_next_slices(self):

        #check if the subslices are exhausted
        if self.cur_sample.slice_num == self.total_slices:
            self.cur_sample.idx += 1
            self.cur_sample.slice_num = 0
            self.cur_sample.complete = (self.cur_sample.idx == self.cur_sample.img.shape[0])

        # if all slices of the current sample are exhausted (continue)
        if self.cur_sample.complete:
            self.img_num += 1

            #reset to the first image if at the end
            if self.img_num == self.last_img + 1:
                print("------------------- NEXT EPOCH -----------------------")
                self.img_num = self.first_img

            img_file = os.path.join(self.root_dir, TRAIN_DIR, IMG_PREFIX + self.files[self.img_num] + EXT)
            label_file = os.path.join(self.root_dir, LABEL_DIR, LABEL_PREFIX + self.files[self.img_num] + EXT) if self.is_labeled else None

            self.cur_sample = Img(self.files[self.img_num], process_image(img_file, self.padding), process_image(label_file, self.padding, False), idx=0, slice_num=0) #img, label, axis, idx


        #calculate the "coords"
        x = self.cur_sample.slice_num % self.num_slices
        y = int((self.cur_sample.slice_num - x) / self.num_slices)

        #print("X: {}, Y: {}".format(x, y))

        x = x * self.slice_stride
        y = y * self.slice_stride

        ex = x + self.slice_size
        ey = y + self.slice_size

        prev_img_slice = self.cur_sample.img[max(self.cur_sample.idx - 1, 0), x:ex, y:ey].astype('float32')
        img_slice = self.cur_sample.img[self.cur_sample.idx, x:ex, y:ey].astype('float32')
        next_img_slice = self.cur_sample.img[min(self.cur_sample.idx + 1, self.cur_sample.img.shape[0] - 1), x:ex, y:ey].astype('float32')
        img_label = self.cur_sample.label[self.cur_sample.idx, x:ex, y:ey] if self.is_labeled else np.array([])

        #if we have an image label filter for the spleen
        if img_label.size != 0:
            tmp_label = (img_label == 0)[np.newaxis, :]

            for nclass in self.classes:
                tmp_label = np.concatenate((tmp_label, (img_label == nclass)[np.newaxis, :]), axis=0)

            img_label = tmp_label

        img_label = tmp_label.astype('float32')

        # move to the next slice
        self.cur_sample.slice_num += 1


        if SAVE_IMAGES:
            im = Image.fromarray(np.uint8(img_slice))
            img_name = "{}_{}_{}".format(self.cur_sample.img_name, self.cur_sample.idx, self.cur_sample.slice_num - 1)
            print("SAVING {}".format(img_name))
            im.save('./gen/gen_' + str(img_name).zfill(4) + ".png")
            im = Image.fromarray(np.uint8(prev_img_slice))
            im.save('./gen/gen_' + str(img_name).zfill(4) + "_prev.png")
            im = Image.fromarray(np.uint8(next_img_slice))
            im.save('./gen/gen_' + str(img_name).zfill(4) + "_next.png")

            for i in range(img_label.shape[0]):
                im = Image.fromarray(np.uint8(img_label[i, :, :])*125)
                im.save('./gen/gen_' + str(img_name).zfill(4) + "_mask_" + str(i) + ".png")


        #update the current object's status
        return prev_img_slice, img_slice, next_img_slice, img_label

    def __len__(self):
        return self.len * (self.num_slices * self.num_slices)

    # TODO: this needs to work based on the index (that's how the iteration begins again!)
    # TODO: it can just reset to previous state
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #get the next slice
        prev_img_slice, img_slice, next_img_slice, img_label = self.get_next_slices()

        # convert to tensors
        imgcs = torch.from_numpy(img_slice).unsqueeze(0)
        imgns = torch.from_numpy(next_img_slice).unsqueeze(0)
        imgps = torch.from_numpy(prev_img_slice).unsqueeze(0)
        mask = torch.from_numpy(img_label) #already has a fourth dimension

        return imgps, imgcs, imgns, mask

