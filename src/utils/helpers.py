import matplotlib.image as mpimg
import numpy as np
import torch
from PIL import Image


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth"""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, patch_size, padding):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    if is_2d:
        padding_channel = ((padding, padding), (padding, padding))
    else:
        padding_channel = ((padding, padding), (padding, padding), (0, 0))
    im = np.lib.pad(im, padding_channel, 'reflect')
    for i in range(padding, imgheight + padding, patch_size):
        for j in range(padding, imgwidth + padding, patch_size):
            if is_2d:
                im_patch = im[j - padding:j + patch_size + padding, i - padding:i + patch_size + padding]
            else:
                im_patch = im[j - padding:j + patch_size + padding, i - padding:i + patch_size + padding, :]
            list_patches.append(im_patch)
    return list_patches


def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels
    
    @param predicted_logits: float32 tensor of shape (batch size, num classes)
    @param reference: int64 tensor of shape (batch_size) with the class number
    """
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()

def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def extract_patches_labels(imgs, gt_imgs, patch_size, padding, n_extract):
    img_patches = [img_crop(imgs[i], patch_size, padding) for i in range(n_extract)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, 0) for i in range(n_extract)]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    # compute label for each patch
    Y = np.asarray([value_to_class(np.mean(gt_patches[i]), 0.25) for i in range(len(gt_patches))])
    return img_patches, Y

def extract_features(img):
    """Extract 6-dimensional features consisting of average RGB color as well as variance"""
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat


def extract_features_2d(img):
    """Extract 2-dimensional features consisting of average gray color as well as variance"""
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


def extract_img_features(filename, patch_size):
    """Extract features for a given image"""
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, 0)
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X


def label_to_img(imgwidth, imgheight, w, h, labels):
    """Convert array of labels to an image"""
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img