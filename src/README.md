# Road Segmentation 

## Description

This repository provides a CNN model for Road Segmentation of Google Maps satellite images. The models performs with a F1-score of 83.3% and an accuracy of 90.3% on the dataset of satellite images from AICrowd website https://www.aicrowd.com/challenges/epfl-ml-road-segmentation.

The dataset used for training and generating the weights (file **weights**) consists of 100 satellites images of size 400x400 each along with their groundtruths.

## Libraries

The code require using the following libraries
- PyTorch 1.10.x
- Torchvision 0.11.x
- Pillow 8.2.x
- Numpy 1.20.xx
- Matplotlib 3.3.x
- scikit-learn 1.0.x
- Python 3.8.x

## How to run the code ?

First, clone the repository. Then, you can generate the predictions on AICrowd dataset of satellite images using the following command line in a shell:

```bash
python run.py
```

If you want to retrain the model from scratch before generating the predictions you can instead run the following command line:

```bash
python run.py --training
```

## File architecture
* **run.py** : the code to train our CNN model and get a csv file with the predictions on AICrowd test set
* **weights** : 
* **data** : folder containing the training set and AICrowd test set
* **utils** : folder containing helper functions
* **cnn.py** : The CNN model and the functions implemented to train our CNN
* **\*.ipynb** : notebooks used to test different ML approaches for the road segmentation task
* **submission.csv** : csv file containing the predictions produced with the CNN on AICrowd test set

## Hardware

- GPU: GeForce RTX 2060
- CPU: intel core i7-9750H
- RAM: 16 GB

## Technical Details

* Patch-based binary classification using CNN (each patch can be classified either as a `road' or as a `background')
* Patches of size 16 x 16 of the original satellite image are classified and then reassembled to constitute the final prediction for the road segmentation
* Use of Data Augmentation of satellite images (vertical and horizontal flips, random rotations (up to +-90°), brightness, contrast, saturation, hue)

## Authors
* Galann Pennec
* Valérian Mangé
* Félix Klein

## References

* How to build a simple CNN model in Pytorch : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
* Data Augmentation of images in PyTorch : https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
* Datasets and DataLoaders in PyTorch : https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec
