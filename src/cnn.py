import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.nn.functional import *

from tqdm import tqdm

from utils.helpers import accuracy


class LeNetModel(nn.Module):
    """CNN For the Road Segmentation Task"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)


    def forward(self, x):
        # Convolutional Layers
        x = self.batch_norm1(leaky_relu(max_pool2d(self.conv1(x), 2)))
        x = self.batch_norm2(leaky_relu(max_pool2d(self.conv2(x), 2)))
        x = self.batch_norm3(leaky_relu(max_pool2d(self.conv3(x), 2)))

        # Flatten
        x = x.view(x.shape[0], -1)

        # Fully Connected Layers
        x = leaky_relu(self.fc1(x))
        x = leaky_relu(self.fc2(x))
        x = leaky_relu(self.fc3(x))
        x = softmax(x, dim=1)

        return x


def train(model, criterion, dataset_train, dataset_valid, optimizer, scheduler, num_epochs, device):
    """
    @param model: torch.nn.Module
    @param criterion: torch.nn.modules.loss._Loss
    @param dataset_train: torch.utils.data.DataLoader
    @param dataset_valid: torch.utils.data.DataLoader
    @param optimizer: torch.optim.Optimizer
    @param num_epochs: int
    """
    print("Starting training")

    loss_train = []
    loss_valid = []

    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        running_loss = 0.
        for batch_x, batch_y in tqdm(dataset_train):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train.append(running_loss / len(dataset_train))


        # Test the quality on the validation set
        model.eval()
        accuracies_valid = []
        f1_valid = []
        running_loss = 0.
        for batch_x, batch_y in dataset_valid:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            running_loss += loss.item()

            accuracies_valid.append(accuracy(prediction, batch_y))
            predicted_labels = np.argmax(prediction.detach().cpu(), 1)
            f1_valid.append(f1_score(batch_y.detach().cpu(), predicted_labels))

        loss_valid.append(running_loss / len(dataset_valid))

        # Printing statistics
        accuracy_epoch = sum(accuracies_valid) / len(accuracies_valid)
        f1_epoch = sum(f1_valid) / len(f1_valid)
        print("Epoch {}".format(epoch + 1))
        print("val accuracy: {:.5f}, f1-score: {:.5f}\n".format(accuracy_epoch, f1_epoch))
        
        scheduler.step(f1_epoch)

    print("Finished")
    return loss_train, loss_valid


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

            predicted_labels = np.argmax(prediction.detach().cpu(), 1)
            f1_train.append(f1_score(batch_y.detach().cpu(), predicted_labels))

        loss_train.append(running_loss / len(dataset_train))
        f1_epoch = sum(f1_train) / len(f1_train)

        # Printing statistics
        print("Epoch {}".format(epoch + 1))
        print("train f1-score: {:.5f}\n".format(f1_epoch))

    print("Finished")
    return


def plot_performance (loss_train, loss_valid):
    """plotting the loss curves for the train and the validation sets"""
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_valid)
    plt.legend(["train loss", "validation loss"])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()


def predict(model, dataset_test):
    """
    @param model: torch.nn.Module
    @param dataset_test: torch.utils.data.DataLoader
    """
    # Running the prediction for the test set
    model.eval()
    predicted_labels = []
    for batch_x in tqdm(dataset_test):
        predicted_logits = model(batch_x).detach().numpy()
        predicted_labels.append(np.argmax(predicted_logits, 1))
    return np.concatenate(predicted_labels)