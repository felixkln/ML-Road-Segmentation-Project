import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import f1_score

import torch
import torchvision.transforms as T

import os, sys, glob, re
from tqdm import tqdm


def train_submissions (model, criterion, dataset_train, optimizer, num_epochs, device):
    print("Starting training")
    model.train()
    
    loss_train = []
    for epoch in range(num_epochs):
        # Train an epoch
        running_loss = 0.
        f1_train = []
        for batch_x, batch_y in tqdm(dataset_train):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            f1_train.append(f1_score(batch_y.detach(), np.argmax(prediction.detach(), 1)))

        loss_train.append(running_loss / len(dataset_train))
        f1_epoch = sum(f1_train) / len(f1_train)

        # Printing statistics
        print("Epoch {}".format(epoch + 1))
        print("train f1-score: {:.5f}\n".format(f1_epoch))

    print("Finished")
    return


def mask_to_submission_strings(model_lenet, image_filename, patch_size, padding, mean_img, std_img):
    """Reads a single image and outputs the strings that should go into the submission file"""
    model_lenet.eval()
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = np.asarray(mpimg.imread(image_filename))
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    patch_size = 16
    for j in range(padding, imgwidth + padding, patch_size):
        for i in range(padding, imgheight + padding, patch_size):
            patch = im[i - padding:i + patch_size + padding, j - padding:j + patch_size + padding, :]
            patch = T.Compose([T.ToTensor(), T.Normalize(mean_img, std_img)])(patch)
            predict = model_lenet(patch[np.newaxis, :])
            label = np.argmax(predict.detach().numpy(), 1)[0]
            yield("{:03d}_{}_{},{}".format(img_number, j - padding, i - padding, label))


def masks_to_submission(model_lenet, submission_filename, image_filenames, patch_size, padding, mean_img, std_img):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in tqdm(image_filenames):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(model_lenet.cpu(), fn, patch_size, padding, mean_img, std_img))