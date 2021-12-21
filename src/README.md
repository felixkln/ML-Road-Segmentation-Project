# Road Segmentation 

## Description  

## Libraries

The code require using the following libraries
- PyTorch
- Numpy 1.17.xx
- Pillow 6.2.xx
- Matplotlib 3.1.x
- Python 3.7.x

## How to run the code ?


 ```bash 
git clone <repo_url> // clone the repo
cd src
python run.py -unet //for Unet architecture (To be used for course evaluation)
#python run.py -normal // for traditional architecutre
  ```


## Technical Details


## Training Hardware

The training was done using Google Colab with the following configuration
- GPU: 1 x NVIDIA Tesla P100 (16GB CoWoS HBM2 at 732 GB/s)
- CPU: 2 vCPU
- RAM: 12 Go

## File overview
* **run.py** : contains the steps to do to run our project and get a csv file submission in the end. In order to use this, type in the command line python3 run.py (-unet or -normal). 
* **unet.h5** : model trained with result F1 = 0.905
* **weights.h5** : weights of the CNN standard trained model with result F1 = 0.882 
* **final_submission.csv** : csv file generated through unet.h5.
* **helpers.py** : contains all the utilities functions used by the neural network.
* **train_xx.py** : contains the training code for the models 

## Authors
* Galann Pennec
* Valérian Mangé
* Félix Klein

## References

* How to build a simple CNN model in Pytorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
* Data Augmentation of images in PyTorch: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
* Datasets and DataLoaders in PyTorch: https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec
