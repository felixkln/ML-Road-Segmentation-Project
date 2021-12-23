import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Transformation of the original images (Data Augmentation)
augmentation = T.Compose([
    lambda x: np.around(x * 255).astype("uint8"), # converts pixels from [0, 1] to [0, 255]
    T.ToPILImage(),
    T.RandomVerticalFlip(0.5),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(90),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    T.RandomAdjustSharpness(sharpness_factor=1.5),
    T.ToTensor(),
])


class TrainsetCNN(Dataset):
    """Class describing the training dataset used by the CNN"""

    def __init__(self, x, y, mean_img, std_img, augmentation=None):
        """Initializing the attributes of the class

        Parameters
        ----------
        x : nd.array
          Input patches
        y : nd.array
          True labels for each patch
        length : int
          Numbers of patches in x
        augmentation : torchvision.transforms.Compose
          Set of augmentations applied to each patch of the dataset

        Returns
        -------
        Initialized TrainsetCNN dataset
        """
        self.x = x
        self.y = torch.tensor(y).long()
        # Number of image patches in the dataset
        self.length = self.x.shape[0]
        # The way we normalize the image patches
        normalization = T.Normalize(mean_img, std_img)
        # The way we define data augmentation in our dataset
        if augmentation:
            self.augmentation = T.Compose([augmentation, normalization])
        else:
            self.augmentation = T.Compose(
                [T.ToTensor(), normalization])  # No Data Augmentation

    def __getitem__(self, idx):
        return self.augmentation(self.x[idx]), self.y[idx]

    def __len__(self):
        return self.length


class TestsetCNN(Dataset):
    """Class describing the test dataset on which we make predictions"""

    def __init__(self, x, mean_train, std_train):
        """Initializing the attributes of the class

        Parameters
        ----------
        x : nd.array
          Test patches
        transformation : torchvision.transforms.Compose
          Normalizing each patch of the test set according to the train dataset
        length : int
          Numbers of test patches in x

        Returns
        -------
        Initialized TestsetCNN dataset
        """
        self.x = x
        self.transformation = T.Compose(
            [T.ToTensor(), T.Normalize(mean_train, std_train)]) # Normalization
        self.length = len(self.x)

    def __getitem__(self, idx):
        return self.transformation(self.x[idx])

    def __len__(self):
        return self.length


def balanced_sampler(labels):
    """Returns a PyTorch sampler for oversampling roads and balancing the dataset

    Parameters
    ----------
    labels : nd.array
      True labels for each patch of the training set

    Returns
    -------
    sampler : torch.utils.data.sampler.WeightedRandomSampler
      A sampler that oversample roads to deal with the imbalanced training dataset

    """
    # defining weights for the classes ('road' and 'background')
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(labels), y=labels)
    print("Dealing with imbalanced datasets")
    print("Background weight: {:.3f}\nRoad weight: {:.3f}".format(
        class_weights[0], class_weights[1]))

    # Picking a road or a background patch with the same probability (=> balanced training set)
    weights = class_weights[labels]
    weights = torch.tensor(weights, dtype=torch.float32)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))
    return sampler
