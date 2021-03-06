import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import re
import argparse
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
from utils.augmented_dataset import *
from utils.submissions import *

# path to training and test datasets
root_dir = "data/training/"
image_test = "data/test_set_images"
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"
# saved weights with a F1-score of 83.2% on test set
path_weights = "weights"

patch_size = 16
window_size = 64 # size of the frame surrounding the patch
padding = (window_size - patch_size) // 2

np.random.seed(42)


def trainCNN(img_patches, Y, mean_img, std_img):
    """Performs the train_submissions function from the cnn.py
    file, by setting fixed parameters beforehand

    Parameters
    ----------
    img_patches : nd.array
      Input images divided into 16x16 patches
    Y : nd.array
      Labels
    mean_img, std_img : nd.array
      Mean and standard deviation for all patches
      on each of the color channels

    Returns
    -------
    model_lenet : LeNetModel
      Model class from cnn.py, which is a subclass of torch.nn.Module
    """

    # DEVICE USED
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # TRAINING THE CNN
    # hyperparameters
    num_epochs = 150
    learning_rate = 1e-3

    # Criterion to optimize
    criterion = torch.nn.CrossEntropyLoss()

    # Building the augmented training set
    batch_size = 200
    sampler = balanced_sampler(Y)
    trainsetAugmented = TrainsetCNN(
        img_patches, Y, mean_img, std_img, augmentation)
    dataset_train = DataLoader(
        trainsetAugmented, batch_size=batch_size, sampler=sampler)

    # Training
    model_lenet = LeNetModel().to(device)
    optimizer = torch.optim.Adam(model_lenet.parameters(), lr=learning_rate)
    train_submissions(model_lenet, criterion, dataset_train,
                      optimizer, num_epochs, device)

    return model_lenet


def run(training):
    """Main function for the whole project
    Loads the training data, optionally retrain the CNN model
    and generate the predictions for AICrowd submission ('submission.csv')

    Parameters
    ----------
    training : bool
        is True if we choose to retrain the CNN model from scratch before the submissions.
        Otherwise our pretrained model generate the submissions with a f1-score of 83.2%

    """
    # LOADING PATCHES AND LABELS
    # Loading set of images
    files = os.listdir(image_dir)
    n = len(files)
    print("Loading " + str(n) + " images...")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    print("Loading " + str(n) + " groundtruths...")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    # Extract patches and labels
    print("\nExtracting patches and labels...")
    img_patches, Y = extract_patches_labels(
        imgs, gt_imgs, patch_size, padding, n)
    # Mean and std of our dataset of image patches
    mean_img = np.mean(img_patches, axis=(0, 1, 2))
    std_img = np.std(img_patches, axis=(0, 1, 2))

    # TRAINING THE CNN IF NECESSARY
    if training:
        # Train the CNN model from scratch
        print("\nTraining the CNN Model from scratch...")
        model_lenet = trainCNN(img_patches, Y, mean_img, std_img)
    else:
        # Load the preexisting model
        print("\nLoading the pretrained CNN Model...")
        model_lenet = LeNetModel()
        model_lenet.load_state_dict(torch.load(path_weights))

    # MAKING THE SUBMISSIONS
    model_lenet.eval()
    print("\nGenerating 'submission.csv'...")
    test_files = list(glob.iglob(image_test + '/*/*.png', recursive=True))
    masks_to_submission(model_lenet.cpu(), "submission.csv",
                        test_files, patch_size, padding, mean_img, std_img)
    print("\nFinished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", dest="training", action="store_true")
    results = parser.parse_args()
    training = results.training is not None
    run(training)
