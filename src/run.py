#### IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os, sys, glob, re
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from PIL import Image
import torch
import torch.nn as nn
from torch.nn.functional import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import *

import torchvision
import torchvision.transforms as T
from torchvision import models

# local imports
from utils.helpers import *
from cnn import *
from dataset import *
from submissions import *

# Choosing the device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



#### LOADING PATCHES AND LABELS
# Loading set of images
root_dir = "data/training/"
image_test = "data/test_set_images"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " groundtruths")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

# Extract patches and labels
patch_size = 16
window_size = 64
padding = (window_size - patch_size) // 2
img_patches, Y = extract_patches_labels(imgs, gt_imgs, patch_size, padding, n)

# Mean and std of our dataset of image patches
mean_img = np.mean(img_patches, axis=(0, 1, 2))
std_img = np.std(img_patches, axis=(0, 1, 2))


#### TRAINING THE CNN
# hyperparameters
num_epochs = 100
learning_rate = 1e-3

# Criterion to optimize
criterion = torch.nn.CrossEntropyLoss()

# Building the augmented training set
batch_size = 200
sampler = balanced_sampler(Y)
trainsetAugmented = TrainsetCNN(img_patches, Y, mean_img, std_img, augmentation)
dataset_train = DataLoader(trainsetAugmented, batch_size=batch_size, sampler=sampler)

# Training the CNN
model_lenet = LeNetModel().to(device)
optimizer = torch.optim.Adam(model_lenet.parameters(), lr=learning_rate)
train_submissions(model_lenet, criterion, dataset_train, optimizer, num_epochs, device)


#### MAKING THE SUBMISSIONS
test_files = list(glob.iglob(image_test + '/*/*.png', recursive=True))
masks_to_submission(model_lenet, "submission.csv", test_files, patch_size, padding, mean_img, std_img)