import os
import random
import re
import torch
import torchvision
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

from os.path import join, isfile
from os import listdir

#temporary imports
import visualize
from visualize import TestVisualizer

def get_data(labels_file, root_dir, transform=None, mode="relative"):
    return PhoneDataset(labels_file=labels_file,
                                   root_dir=root_dir, transform=transform, mode=mode)


def read_txt(root_dir, labels_file):

    if isfile(root_dir):
        files = [root_dir.split('/')[-1]]
    else:
        files = [f for f in os.listdir(root_dir) if isfile(join(root_dir, f))]

    labels = [ [line.split(' ',1)[0], float(line.split(' ',2)[1]), float(line.split(' ',2)[2])] for line in open(labels_file) if (line.split(' ',1)[0] in files)]
    labels = sorted(labels, key=lambda tup: int(tup[0].split('.',1)[0]))

    if len(labels) == 0:
        labels = [ [f.split(' ', 1)[0], float(0.0), float(0.0)] for f in files ]

    return np.array(labels)

class PhoneDataset(Dataset):
    def __init__(self, labels_file, root_dir, transform=None, mode="relative"):
        self.labels = read_txt(root_dir, labels_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                               self.labels[idx][0])
        image = io.imread(img_name)

        # numpy is W x H x C
        # torch is C x H x W
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        target = torch.from_numpy(self.labels[idx, 1:3].astype('float').reshape(-1,2).squeeze())

        if self.transform:
            image = self.transform(image)

        if self.mode == "absolute":
            target[0] *= image.size()[2]
            target[1] *= image.size()[1]

        return image, target

    def get_labels(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()

         img_name = os.path.join(self.root_dir,
                                self.labels[idx][0])
         image = io.imread(img_name)

         # numpy is W x H x C
         # torch is C x H x W
         image = torch.from_numpy(image.transpose((2, 0, 1)))
         target = self.labels[idx, 1:3].astype('float').reshape(-1,2)

         return img_name, image, target

    def find(self, tensor, values):
        return torch.nonzero(tensor[..., None] == values)


#creates new images in a new directory
#creates a corresponding labels file
class DatasetBuilder(object):
    def __init__(self, phone_dataset, result_dir, new_labels_file, PILtransforms, generate=1, overwrite=True):
        self.PILtransforms = PILtransforms
        self.dataset = phone_dataset
        self.result_dir = result_dir+'/' if result_dir[-1] != '/' else result_dir
        self.new_labels_file = new_labels_file
        self.gen = generate
        self.overwrite = overwrite

         #ensures the creation of the directory
        dirs = re.split("/", result_dir)
        directory = ""

        for i in range(len(dirs)):
            directory += dirs[i] + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)

    def generate(self):
        #create labels file

        if not self.overwrite:
            status = "a"
        else:
            status="w"

        labels_file = open(self.new_labels_file, status)

        #iterate over the dataset and generate the new samples
        for s_num in range(len(self.dataset)):
            for i in range(self.gen):
                img_name, image, label = self.dataset.get_labels(s_num)

                #let the first sample through with no transforms
                if i != 0:
                    sample_image, pil_image, new_label = self.exec_PIL_transforms(image, label)
                else:
                    sample_image, pil_image, new_label = img_name, torchvision.transforms.functional.to_pil_image(image), label

                file_name = self.result_dir+re.split('[/.]', img_name)[-2]+"_"+str(i)+"."+img_name.split('.')[-1]
                print(file_name)
                pil_image.save(file_name)

                #write the value to the labels file
                labels_file.write((file_name.split('/')[-1]+" {:.4f} "+" {:.4f}"+"\n").format(new_label[0,0].item(), new_label[0,1].item()))

        labels_file.close()

    # label is a numpy array
    def exec_PIL_transforms(self, image, label, image_name = None):
        #transform from tensor to PIL image
        transformed_sample = torchvision.transforms.functional.to_pil_image(image)
        transformed_label_image = torchvision.transforms.functional.to_pil_image(torch.zeros(image.size(), dtype=torch.uint8))
        prev_label = label.copy()
        safety_pixels = 841 #29*29

        ## compute coords that need to be to 1
        ## label coords to PIL coords
        coordx = int(image.size()[2]*label[0][0])
        coordy = int(image.size()[1]*label[0][1])

        #safe box for the phone (only in training set => so it stays in bounds yikes)
        dist = lambda x,y: pow(pow(x[0]-y[0], 2) + pow(x[1]-y[1],2), 0.5) / 9.9

        for i in range(29):
            for j in range(29):
                x = [coordx-14+i, coordy-14+j]
                y = [coordx, coordy]
                transformed_label_image.putpixel((coordx-14+i, coordy-14+j), (int(255*(1-dist(x,y))), 0, 0))

        transformed_label_image.putpixel((coordx, coordy), (255, 255, 255));
        transforms = self.PILtransforms

        if self.PILtransforms != None:
            if(random.randint(0,1000) > 500):
                transforms.reverse()
            for tsfrm in transforms:
                transformed_sample, transformed_label_image, prev_label, safety_pixels = tsfrm(transformed_sample, transformed_label_image, prev_label, safety_pixels)

        #transform it
        label_image = torchvision.transforms.functional.to_tensor(transformed_label_image)
        sample_image = torchvision.transforms.functional.to_tensor(transformed_sample)

        new_label = torch.from_numpy(prev_label)

        return sample_image, transformed_sample, new_label
