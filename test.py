# %% -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr
from tqdm import tqdm
from config import *
import numpy as np
import torchvision

from losses.plain_dice import dice_coeff

from losses.seg_losses import DiceLoss
from load_data import SpleenDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.unets import UNet
from models.deeplabv3 import DeepLabV3
from PIL import Image

parser = argparse.ArgumentParser(
     description='UNet + BDCLSTM for BraTS Dataset')
parser.add_argument('--model_folder', type=str, default=None, metavar='str',
                    help='model folder to load')
parser.add_argument('--data_folder', type=str, default='./Data/', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--test_img_range', type=int, nargs=2, default=[1, 20], help='Image range for train')

args = parser.parse_args()

DATA_FOLDER = args.data_folder
SAVE_TEST_IMAGES = False

print(args.test_img_range)
mrange = tuple(args.test_img_range)


def save_images(img, sample_num, counter, multiplier=1.0, real_img=False, tag=''):
    print("SAVING IMAGES")
    for i in range(img.size()[0]):

        new_arr = (img[i, int(not real_img), :, :]*multiplier).cpu().byte()

        #print("MAX: {} MIN {}".format(torch.max(new_arr).item(), torch.min(new_arr).item()))
        #nparray = np.uint8(new_arr.numpy()) #this seems to work
        #im = Image.fromarray(nparray)

        pil_img = torchvision.transforms.functional.to_pil_image(new_arr, mode='L')
        pil_img.save('./gen/'+ tag + '_img_'+ str(i+1) + '_' + str(counter) + '.png')

for i in range(mrange[0], mrange[1]+1):
    dset_test = dset_train = SpleenDataset(DATA_FOLDER, tuple([i, i]), SLICE_SIZE, 80, 5, classes=CLASSES, skew_start=0.0, threshold=0.005) #will this fail due to different size?
    test_loader = DataLoader(dset_train, batch_size=1, num_workers=1)

    # load the model
    model = DeepLabV3(num_classes = len(CLASSES) + 1)
    model.load_state_dict(torch.load(args.model_folder))
    model.cuda()
    model.eval()

    side_length = 512 + dset_test.padding

    # initialize our holder and our counter
    mimage = torch.Tensor()

    counter = 0
    for i in range(len(dset_test)):
        with torch.no_grad():

            # a new function we implement "fetch_tensor"
            image1, image2, image3, mask, coords = dset_test.fetch_tensor(i)
            image2, mask = torch.cat((image1, image2, image3), dim=0).unsqueeze_(0).cuda(),mask.unsqueeze_(0).cuda()

            #output = image2
            output = torch.nn.functional.sigmoid(model(image2)) # force between 0 and 1
            output = (output.round() > 0.5).int()

            print(output.size())
            print(mask.size())

            # if the counter is divisible by 3
            if counter % 9 == 10:
                # new slice
                mslice = torch.Tensor([side_length, side_length])

            # calculate the upper left quadrant for the image (look at load_data code) => actually store here (maybe add a new function that returns coords => part of fetch?)


            # let us try saving it :)
            if SAVE_TEST_IMAGES:
                save_images(output, i, counter, 255.0)
                save_images(mask, i, counter, 255.0, tag='mask')

            counter += 1

