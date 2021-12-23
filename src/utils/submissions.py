import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import f1_score

import torch
import torchvision.transforms as T

import os
import sys
import glob
import re
from tqdm import tqdm


def train_submissions(model, criterion, dataset_train, optimizer, num_epochs, device):
    """Performs the optimization of the training loss
    so as to generate the .csv file for submission afterwards

    Parameters
    ----------

    model : torch.nn.Module
      Model used for the prediction (a CNN for instance)
    criterion : torch.nn.modules.loss._Loss
      Type of loss to optimize (CrossEntropyLoss for instance)
    dataset_train : torch.utils.data.DataLoader
      Loaded training dataset
    optimizer : torch.optim.Optimizer
      Type of method used to minimize the loss (GD, SGD, ADAM, ISTA, FISTA, ADAGRAD, etc.)
    num_epochs : int
      Number of iterations over the total dataset
    device : torch.cuda
      If available, this function uses the GPU of your computer to accelerate the running time

    """
    print("Starting training")
    model.train()

    loss_train = []
    for epoch in range(num_epochs):
        # Train an epoch
        running_loss = 0. # training loss accumulated for one epoch
        f1_train = []
        for batch_x, batch_y in tqdm(dataset_train):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            running_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the f1-score for the current training batch
            predicted_labels = np.argmax(prediction.detach().cpu(), 1)
            f1_train.append(f1_score(batch_y.detach().cpu(), predicted_labels))

        # add the averaged training loss and f1-score for this epoch
        loss_train.append(running_loss / len(dataset_train))
        f1_epoch = sum(f1_train) / len(f1_train)

        # Printing statistics
        print("Epoch {}".format(epoch + 1))
        print("train f1-score: {:.5f}\n".format(f1_epoch))

    print("Finished")
    return


def mask_to_submission_strings(model_lenet, image_filename, patch_size, padding, mean_img, std_img):
    """Reads a single image and outputs the strings that should go into the submission file

    Parameters
    ----------
    model_lenet : torch.nn.Module
      Model used for the prediction
    image_filename : string
      Path, name and location of the image
    patch_size : int
      Size of the patches (cropped images)
    padding : int
      Shift value for the cropping operation
    mean_img, std_img : nd.array
      Mean and standard deviation for each channel

    Returns
    -------
    Strings representing each line : number of image, number of patch and corresponding prediction
    """
    model_lenet.eval()
    img_number = int(re.search(r"\d+", image_filename).group(0))
    # Load the test image
    im = np.asarray(mpimg.imread(image_filename))
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    # Add padding to the image for extracting a frame surrounding each patch
    im = np.lib.pad(
        im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    patch_size = 16
    # Cropping the test satellite image
    for j in range(padding, imgwidth + padding, patch_size):
        for i in range(padding, imgheight + padding, patch_size):
            patch = im[i - padding:i + patch_size + padding,
                       j - padding:j + patch_size + padding, :]
            # Normalizing each patch
            patch = T.Compose(
                [T.ToTensor(), T.Normalize(mean_img, std_img)])(patch)
            # Compute the prediction for this patch
            predict = model_lenet(patch[np.newaxis, :])
            label = np.argmax(predict.detach().numpy(), 1)[0]
            # Add the submission line associated to the prediction for this patch
            yield("{:03d}_{}_{},{}".format(img_number, j - padding, i - padding, label))


def masks_to_submission(model_lenet, submission_filename, image_filenames, patch_size, padding, mean_img, std_img):
    """Converts test images into a submission file for AICrowd website
    Generates the submission file 'submission.csv' containing all the predictions corresponding to each patch of each test image

    Parameters
    ----------
    model_lenet : torch.nn.Module
      Model used for the prediction
    submission_filename : string
      Name of the output file
    image_filename : string
      Path, name and location of the image
    patch_size : int
      Size of the patches (cropped images)
    padding : int
      Shift value for the cropping operation
    mean_img, std_img : nd.array
      Mean and standard deviation of each channel

    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in tqdm(image_filenames):
            # Generate the prediction lies for one satellite image
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(
                model_lenet.cpu(), fn, patch_size, padding, mean_img, std_img))
