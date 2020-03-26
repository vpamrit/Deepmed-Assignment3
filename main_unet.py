
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
from losses.other_losses import GDiceLoss
from losses.seg_losses import DiceLoss
from losses.combined_loss import CombinedLoss

from load_data import SpleenDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.unets import UNet
from models.deeplabv3 import DeepLabV3


# %% import transforms

# %% Training settings
parser = argparse.ArgumentParser(
    description='UNet + BDCLSTM for BraTS Dataset')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--size', type=int, default=128, metavar='N',
                    help='imsize')
parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--data-folder', type=str, default='./Data/', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save', type=str, default='OutMasks', metavar='str',
                    help='Identifier to save npy arrays with')
parser.add_argument('--modality', type=str, default='flair', metavar='str',
                    help='Modality to use for training (default: flair)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='str')
parser.add_argument('--train_img_range', type=int, nargs=2, default=[0, 24], help='Image range for train')
parser.add_argument('--valid_img_range', type=int, nargs=2,  default=[25, 25], help='Image range for train')
parser.add_argument('--load_model', type=str, default=None, help='Image range for train')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

DATA_FOLDER = args.data_folder
THRESHOLD = 0.006

# %% Loading in the Dataset
dset_train = SpleenDataset(DATA_FOLDER, tuple(args.train_img_range), SLICE_SIZE, 150, 3, classes=CLASSES, threshold=THRESHOLD)
dset_valid = SpleenDataset(DATA_FOLDER, tuple(args.valid_img_range), SLICE_SIZE, 150, 3, classes=CLASSES, threshold=THRESHOLD)

train_loader = DataLoader(dset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
valid_loader = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=4, drop_last=True)


print("Training Data : ", len(train_loader.dataset))
print("Validation Data :", len(valid_loader.dataset))

# %% Loading in the model
num_classes = len(CLASSES)+1
model = DeepLabV3(num_classes = num_classes) #UNet(num_classes=num_classes)

if args.load_model != None:
    model.load_state_dict(torch.load(args.load_model))

V3 = True

model.cuda()

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.99)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))


# Defining Loss Function
criterion = CombinedLoss([torch.nn.BCEWithLogitsLoss(), DiceLoss(weight=torch.FloatTensor(WEIGHTS))]) #torch.nn.BCEWithLogitsLoss() #DiceLoss(weight=torch.FloatTensor(WEIGHTS))


def train(epoch, loss_list, counter):
    model.train()
    for batch_idx, (image1, image2, image3, mask) in enumerate(train_loader):
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()
        if V3:
            #combine all three
            image2 = torch.cat([image1, image2, image3], dim=1)

        optimizer.zero_grad()
        output = model(image2)

        #print("Mask size is {} Output size is {}".format(mask.size(), output.size()))
        #need to ignore padding border here (for loss)
        loss = criterion(output, mask)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage DICE Loss: {:.6f}'.format(
                 epoch, batch_idx * len(image2), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))


    #here we can compute the average dice coefficient on the remaining dataset
    model.eval()

    total_loss = 0
    dice_total = 0
    count = 0

    print('Computing validation loss...')

    #for assembling 3d for dice
    full_mask = torch.Tensor()
    full_out = torch.Tensor()

    # N x C x H x W
    # N x C x H x W x D
    for batch_idx, (image1, image2, image3, mask) in enumerate(valid_loader):
        with torch.no_grad():
            image1, image2, image3, mask = image1.cuda(), image2.cuda(), image3.cuda(), mask.cuda()

            if V3:
                image2 = torch.cat([image1, image2, image3], dim=1)

            output = model(image2)

            loss = criterion(output, mask)
            total_loss += loss.item()

            #force the output to 0 and 1
            #construct full 3D tensor for dice calculation
            #turn off cuda here

            pure_output = (output[:,1,:,:].clone().detach().round() > 0).float()
            _3dmask = mask[:, 1, :, :].clone().detach()

            #force to dim 5 N x C x H x W x D
            _3dmask = _3dmask.permute((1, 2, 0)).unsqueeze_(0).unsqueeze_(0)
            _3doutput = pure_output.permute((1, 2, 0)).unsqueeze_(0).unsqueeze_(0)

            if full_mask.size()[0] == 0:
                full_mask = _3dmask
                full_out = _3doutput
            else:
                full_mask = torch.cat((full_mask, _3dmask), 4)
                full_out = torch.cat((full_out, _3doutput), 4)


            #lame way to calculate the dice
            dice_v = dice_coeff(pure_output, mask[:, 1, :, :])
            dice_total += dice_v if dice_v < 0.9 else 0
            count += 1 if dice_v < 0.9 else 0

            if SAVE_VALID_IMAGES and epoch in SAVE_EPOCHS:
                print("SAVING IMAGES")
                for i in range(output.size()[0]):
                    pil_img = torchvision.transforms.functional.to_pil_image((output[i, 1, :, :]*125).  squeeze_().cpu())
                    mask_img = torchvision.transforms.functional.to_pil_image((mask[i, 1, :, :]*125).   squeeze_().cpu())

                    pil_img.save('./gen/gen_img_' + str(counter) + '.png')
                    mask_img.save('./gen/mask_img_' + str(counter) + '.png')
                    counter += 1

    avg_loss = total_loss / len(valid_loader.  dataset)
    avg_dice = dice_total / (max(1, count)) #divide by num batches
    avg_3D_dice = dice_coeff(full_mask, full_out)

    loss_list[0].append(avg_loss)
    loss_list[1].append(avg_dice)
    loss_list[2].append(avg_3D_dice)

    print('Validation Epoch:  Avg Loss {}\n'.format(avg_loss))
    print('Dice Coeff Avg {}'.format(avg_dice))
    print('Full 3D Dice Result {}'.format(avg_3D_dice))


def save_create_plot(loss_list, plot_name):
    plt.plot(loss_list)
    plt.title("UNet bs={}, ep={}, lr={}".format(args.batch_size,
                                                i, args.lr))
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.savefig("./{}_Loss_bs={}_lr={}.png".format(plot_name, args.batch_size, args.lr))
    plt.clf()


loss_list = [[1.2], [1.2], [1.2]]
counter = 0


## main function
os.makedirs(SAVE_DIR, exist_ok=True)


for i in tqdm(range(args.epochs)):
    train(i, loss_list, counter)
    torch.save(model.state_dict(), SAVE_DIR + 'deeplabv3-final-{}.pth'.format(i))

    if (i+1) % 7 == 0:
        THRESHOLD = THRESHOLD / 2 if THRESHOLD >= 0.003 else 0.00
        dset_train.clean(THRESHOLD)
        dset_valid.clean(THRESHOLD)


    counter += 1

    # overwrite plot at the end of each iteration
    save_create_plot(loss_list[0], 'Avg_Loss')
    save_create_plot(loss_list[1], '2D_ Dice')
    save_create_plot(loss_list[2], '3D_Dice')

