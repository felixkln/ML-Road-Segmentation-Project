# Road Segmentation 

## Description

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

First, clone the repository. Then, you can train the CNN model and generate the predictions on AICrowd dataset of satellite images using the following command line in a shell:

```bash
python run.py
```


## Technical Details


## Training Hardware

- GPU: GeForce RTX 2060
- CPU: intel core i7-9750H
- RAM: 16 GB

## File architecture
* **submission.csv** : csv file containing the predictions produced with the CNN on AICrowd test set
* **train_xx.py** : contains the training code for the models
* **run.py** : the code to train our CNN model and get a csv file with the predictions on AICrowd test set
* **data** : folder containing the training set and AICrowd test set
* **utils** : folder containing the helper functions
* **\*.ipynb***

## Authors
* Galann Pennec
* Valérian Mangé
* Félix Klein

## References

* How to build a simple CNN model in Pytorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
* Data Augmentation of images in PyTorch: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
* Datasets and DataLoaders in PyTorch: https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec
