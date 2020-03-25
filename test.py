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

from plain_dice import dice_coeff

from seg_losses import DiceLoss
from load_data import SpleenDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.unets import UNet
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
SAVE_TEST_IMAGES = True

print(args.test_img_range)
mrange = tuple(args.test_img_range)


def save_images(img, sample_num, counter, multiplier=1.0, real_img=False, tag=''):
    print("SAVING IMAGES")
    for i in range(img.size()[0]):

        new_arr = (img[i, int(not real_img), :, :]*multiplier).cpu().byte()

        print("MAX: {} MIN {}".format(torch.max(new_arr).item(), torch.min(new_arr).item()))
        #nparray = np.uint8(new_arr.numpy()) #this seems to work
        #im = Image.fromarray(nparray)

        pil_img = torchvision.transforms.functional.to_pil_image(new_arr, mode='L')


        pil_img.save('./gen/'+ tag + '_img_'+ str(i+1) + '_' + str(counter) + '.png')

for i in range(mrange[0], mrange[1]+1):
    dset_test = dset_train = SpleenDataset(DATA_FOLDER, tuple([i, i]), SLICE_SIZE, 80, 5,classes=CLASSES) #will this fail due to different size?
    test_loader = DataLoader(dset_train, batch_size=1, num_workers=1)

    # load the model
    model = UNet(num_classes = len(CLASSES) + 1)
    model.load_state_dict(torch.load(args.model_folder))
    model.cuda()
    model.eval()

    counter = 0
    for batch_idx, (image1, image2, image3, mask) in enumerate(test_loader):
        with torch.no_grad():
            image1, image2, image3, mask = image1.cuda(), image2.cuda(), image3.cuda(), mask.cuda()
            #output = image2
            output = model(image2)

            # let us try saving it :)
            if SAVE_TEST_IMAGES:
                save_images(output, i, counter, 255.0, real_img=True)
                save_images(mask, i, counter, 255.0, tag='mask')
                counter += 1
