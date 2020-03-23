
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

from losses import DICELossMultiClass, DICELoss
from seg_losses import DiceLoss
from load_data import SpleenDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import UNet


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
parser.add_argument('--optimizer', type=str, default='SGD', metavar='str',
                    help='Optimizer (default: SGD)')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

DATA_FOLDER = args.data_folder

# %% Loading in the Dataset
slice_size = 240

# %% Loading in the Dataset
dset_train = SpleenDataset(DATA_FOLDER, (1, 15), slice_size, 80, 5, classes=CLASSES) #will this fail     due to different size?
dset_valid = SpleenDataset(DATA_FOLDER, (0, 0), slice_size, 160, 3, classes=CLASSES)

train_loader = DataLoader(dset_train, batch_size=args.batch_size, num_workers=4)
valid_loader = DataLoader(dset_valid, batch_size=args.batch_size, num_workers=4)


print("Training Data : ", len(train_loader.dataset))
print("Validation Data :", len(valid_loader.dataset))

# %% Loading in the model
model = UNet(num_classes=len(CLASSES)+1)
model.cuda()

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.99)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))


# Defining Loss Function
criterion = DiceLoss()


def train(epoch, loss_list, counter):
    model.train()
    for batch_idx, (image1, image2, image3, mask) in enumerate(train_loader):
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()

        image1, image2, image3, mask = Variable(image1), \
            Variable(image2), \
            Variable(image3), \
            Variable(mask)

        optimizer.zero_grad()

        output = model(image2)

        #print("Mask size is {} Output size is {}".format(mask.size(), output.size()))
        #need to ignore padding border here (for loss)
        padding = dset_train.padding
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

    print('Validation Epoch: Loss {}, Avg Loss {}\n'.format(total_loss, total_loss / len(valid_loader.  dataset)))
    print('Dice Coeff Avg {}'.format(dice_total / (max(1, count)))) #divide by num batches
    print('Full 3D Dice Result {}'.format(dice_coeff(full_mask, full_out)))



loss_list = []
counter = 0

## main function
os.makedirs(SAVE_DIR, exist_ok=True)

for i in tqdm(range(args.epochs)):
    train(i, loss_list, counter)
    torch.save(model.state_dict(), SAVE_DIR + 'unet-final-{}'.format(i))

    counter += 1

plt.plot(loss_list)
plt.title("UNet bs={}, ep={}, lr={}".format(args.batch_size,
                                            args.epochs, args.lr))
plt.xlabel("Number of iterations")
plt.ylabel("Average DICE loss per batch")
plt.savefig("./plots/{}-UNet_Loss_bs={}_ep={}_lr={}.png".format(args.save,
                                                                args.batch_size,
                                                                args.epochs,
                                                                args.lr))
