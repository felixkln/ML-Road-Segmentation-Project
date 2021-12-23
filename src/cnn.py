import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.nn.functional import *

from tqdm import tqdm

# Local import
from utils.helpers import accuracy


class LeNetModel(nn.Module):
    """CNN For the Road Segmentation Task"""

    def __init__(self):
        """Initializes the layers of the convolutional network

        Parameters
        ----------
        conv1, conv2, conv3 : torch.nn.Conv2d
          Convolutional layers from torch.nn
        batch_norm1, batch_norm2, batch_norm3 : torch.nn.BatchNorm2d
          Batch normalizing layers from torch.nn
        fc1, fc2, fc3 : torch.nn.Linear
          Fully connected layers from torch.nn
          Put at the end of the network before the softmax classifier

        Returns
        -------
        Initialized LeNetModel object

        """
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
        """Performs a prediction on input x by feeding it into
        the model to obtain the output

        Parameters
        ----------
        x : torch.Tensor
          Input data to be fed into the network

        Returns
        -------
        x : torch.Tensor
          The prediction estimated from input data,
          returned as a distribution over the two classes ('road' and 'background').

        """
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
    """Performs the training and the validation over several epochs.
    Print the validation accuracy and f1-score of each epoch trained.

    Parameters
    ----------

    model : torch.nn.Module
      Model used for the prediction (a CNN for instance)
    criterion : torch.nn.modules.loss._Loss
      Type of loss to optimize (CrossEntropyLoss for instance)
    dataset_train : torch.utils.data.DataLoader
      Loaded training dataset
    dataset_valid : torch.utils.data.DataLoader
      Loaded validation dataset, generated from the original training set
    optimizer : torch.optim.Optimizer
      Type of method used to minimize the loss (GD, SGD, ADAM, ISTA, FISTA, ADAGRAD, etc.)
    num_epochs : int
      Number of iterations over the total dataset
    device : torch.cuda
      If available, this function uses the GPU of your computer to accelerate the running time

    Returns
    -------

    loss_train, loss_valid: float
      training and validation losses
    """
    print("Starting training")

    loss_train = []
    loss_valid = []

    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        running_loss = 0. # training loss accumulated for one epoch
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

        # add the averaged training loss for this epoch
        loss_train.append(running_loss / len(dataset_train))

        # Test the quality on the validation set
        model.eval()
        accuracies_valid = []
        f1_valid = []
        running_loss = 0.
        for batch_x, batch_y in dataset_valid:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Compute the loss for the current validation example
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            running_loss += loss.item()

            # Add the accuracy and the f1-score for the current validation example
            accuracies_valid.append(accuracy(prediction, batch_y))
            predicted_labels = np.argmax(prediction.detach().cpu(), 1)
            f1_valid.append(f1_score(batch_y.detach().cpu(), predicted_labels))

        # add the averaged validation loss for this epoch
        loss_valid.append(running_loss / len(dataset_valid))

        # Printing statistics for this epoch
        accuracy_epoch = sum(accuracies_valid) / len(accuracies_valid)
        f1_epoch = sum(f1_valid) / len(f1_valid)
        print("Epoch {}".format(epoch + 1))
        print(
            "val accuracy: {:.5f}, f1-score: {:.5f}\n".format(accuracy_epoch, f1_epoch))

        scheduler.step(f1_epoch)

    print("Finished")
    return loss_train, loss_valid


def plot_performance(loss_train, loss_valid):
    """plotting the loss curves for the train and the validation sets
    x axis : epoch
    y axis : training and validation loss curves

    Parameters
    ----------

    loss_train, loss_valid : float
      loss of the training and validation datasets generated by the train function

    """
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_valid)
    plt.legend(["train loss", "validation loss"])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()


def predict(model, dataset_test):
    """From the test set, this function computes (predicts)
    the output (labels)

    Parameters
    ----------
    model : torch.nn.Module
      Model used for the prediction (a CNN for instance)
    dataset_test : torch.utils.data.DataLoader
      The test set used to compute its labels

    Returns
    -------
    predicted_labels : nd.array
      Output of the model, and prediction
    """
    # Running the prediction for the test set
    model.eval()
    predicted_labels = []
    for batch_x in tqdm(dataset_test):
        predicted_logits = model(batch_x).detach().numpy()
        # Labels are predicted by taking the maximum of the probability distribution over the two classes
        predicted_labels.append(np.argmax(predicted_logits, 1))
    return np.concatenate(predicted_labels)
