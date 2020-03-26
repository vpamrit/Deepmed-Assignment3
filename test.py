# %% -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import nibabel as nib
import os
import statistics as stat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr
import config

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
from config import *

parser = argparse.ArgumentParser(
     description='UNet + BDCLSTM for BraTS Dataset')
parser.add_argument('--model_folder', type=str, default=None, metavar='str',help='model folder to load')
parser.add_argument('--test_img_range', type=int, nargs=2, default=[1, 20], help='Image range for train')
parser.add_argument('--data_folder', type=str, default='./data/Training/', help='Image range for test')

args = parser.parse_args()

DATA_FOLDER = args.data_folder
SAVE_TEST_IMAGES = False #True
SAVE_TEST_3D = True
TEST_SAVE_DIR = './results/'

print(args.test_img_range)
mrange = tuple(args.test_img_range)


# load the model
model = DeepLabV3(num_classes = len(CLASSES) + 1)
model.load_state_dict(torch.load(args.model_folder))
model.cuda()
model.eval()

def save_images(img, subject_num, slice_depth, multiplier=1.0, real_img=False, tag=''):
    print("SAVING IMAGES")
    #for i in range(img.size()[0]):

    new_arr = (img[:, :]*multiplier).cpu().byte()

        #print("MAX: {} MIN {}".format(torch.max(new_arr).item(), torch.min(new_arr).item()))
        #nparray = np.uint8(new_arr.numpy()) #this seems to work
        #im = Image.fromarray(nparray)

    pil_img = torchvision.transforms.functional.to_pil_image(new_arr, mode='L')
    pil_img.save('./gen/'+ tag + '_img_' + str(subject_num) + '_' + str(slice_depth) + '.png')


def get_angry(mten):
    if mten.size()[0] == 0:
        return

    value = torch.max(mten).item()
    print("MAX IS {}".format(value))
    assert value <= 1.0, "Max value greater than one!"

    return


def process_tensor(mten):
    value = torch.max(mten).item()

    return np.uint8(np.transpose(mten.cpu().numpy(), (1,2,0)))

# start part

dice_vals = []

for sub_num in range(mrange[0], mrange[1]+1):
    dset_test = SpleenDataset(DATA_FOLDER, tuple([sub_num, sub_num]), SLICE_SIZE, 150, 3, classes=CLASSES, skew_start=0.0, threshold=0.0)
    dset_pure = SpleenDataset(DATA_FOLDER, tuple([sub_num, sub_num]), 512, 512, 1, classes=CLASSES, skew_start=0.0, threshold=0.0)


    padding = dset_test.padding
    side_length = 512 + padding*2

    # initialize our holder and our counter
    mcount = torch.Tensor()

    counter = 0
    slice_depth = 0
    mslice = torch.zeros([side_length, side_length]).cuda()
    mcount = torch.zeros([side_length, side_length]).cuda()

    # place to store the 3D volume
    all_slices = torch.Tensor()
    all_masks = torch.Tensor()


    for i in range(len(dset_test)):
        with torch.no_grad():
            # a new function we implement "fetch_tensor"
            image1, image2, image3, mask, coords = dset_test.fetch_tensor(i)
            image2, mask = torch.cat((image1, image2, image3), dim=0).unsqueeze_(0).cuda(),mask.unsqueeze_(0).cuda()

            #fetch actual images

            #output = image2
            output = torch.nn.functional.sigmoid(model(image2)) # force between 0 and 1
            #output = #$(output.round() > 0.5).int()

            # if the counter is divisible by 3


            # use the coords to set the part of the image
            cut_output = output[0, 1, :, :]
            mslice[coords[0]:coords[1], coords[2]:coords[3]] += cut_output
            mcount[coords[0]:coords[1], coords[2]:coords[3]] += torch.ones(cut_output.size()).cuda()

            if counter+1 == dset_test.total_slices:
                # get the mask from pure dataset
                image1p, image2p, image3p, maskp, coordsp = dset_pure.fetch_tensor(slice_depth)

                if dset_pure.is_labeled:
                    maskp = maskp[1, :, :]

                    #mask 3D
                    prepped_mask = maskp.clone().unsqueeze_(0) # 1 x H x W
                    all_masks = torch.cat([all_masks, prepped_mask], dim=0) if all_masks.size()[0] else prepped_mask # D x H x W

                #remove padding
                mslice = mslice[padding:-padding, padding:-padding]
                mcount = mcount[padding:-padding, padding:-padding]


                # divide it up
                #process mslice here as desired
                mslice = mslice / mcount
                mslice = (mslice > 0.5).float()

                #add to 3D volume
                prepped_mslice = mslice.clone().unsqueeze_(0) # 1 x H x W
                all_slices = torch.cat([all_slices, prepped_mslice], dim=0) if all_slices.size()[0] else prepped_mslice # D x H x W


                #compare maskp and mslice
                if SAVE_TEST_IMAGES:
                    save_images(maskp, sub_num, slice_depth, 255.0, tag='mask')
                    save_images((mslice / mcount), sub_num, slice_depth, 255.0)


                # initialize new slice
                mslice = torch.zeros([side_length, side_length]).float().cuda()
                mcount = torch.zeros([side_length, side_length]).float().cuda()
                counter = -1
                slice_depth += 1

            counter += 1


    if SAVE_TEST_3D:
    # since the dataset is of size one
        img_name = dset_pure.samples[0].img_name
        print("SAVING {}".format(img_name))

        og_file = os.path.join(DATA_FOLDER, TRAIN_DIR, IMG_PREFIX + img_name + EXT)

        # load the original file
        img = nib.load(og_file)

        # load the image
        new_img = nib.Nifti1Image(process_tensor(all_slices), img.get_affine(), img.get_header())


        out_file = TEST_SAVE_DIR + LABEL_PREFIX + img_name + '.nii.gz'
        nib.save(new_img, out_file)


    if dset_pure.is_labeled:
        dice = dice_coeff(all_slices.unsqueeze_(0).cuda(), all_masks.unsqueeze_(0).cuda()).cpu()
        mdice = dice.numpy()[0]

        print("DICE IS {}".format(mdice))

        dice_vals.append(mdice)


print("Dice vals {}\n".format(dice_vals))
#print("Std dev {} Mean {} Median {}".format(stat.stdev(dice_vals), stat.mean(dice_vals), stat.median(dice_vals)))
print("COMPLETE")

