import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import sys
import tensorflow as tf
from PIL import Image

img_path = "E:/machine_learning/HumanHacking_project/train2_images"
train_path = "E:/machine_learning/HumanHacking_project/train.csv"
mask_path = "E:/machine_learning/HumanHacking_project/masks"

train = pd.read_csv(train_path)
img_size = 3000


def resize_img(img):
    return cv2.resize(img, [img_size, img_size])


def rle2mask(mask_rle, shape=(img_size, img_size)):
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
	    img[lo:hi] = 1
	return img.reshape(shape).T

def mask2rle(img):

    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def read_image(image_id, negative):
	image = cv2.imread(f'{img_path}/{image_id}.jpg')

	if len(image.shape) == 5:
		image = image.squeeze().transpose(1, 2, 0)

	if negative:
		image = image - image.min()
		image = image / (image.max() - image.min())
		image = image * 255
		image = 255 - image.astype(np.uint8)

	image = resize_img(image)

	return image

def show_image_and_masks(rows=4, cols=3):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*8, rows*6))
    image_ids = train['id']
    
    for r in range(rows):
        image_id = image_ids[r]
        df_row = train.loc[train['id'] == image_id].head(1).squeeze()
        image = read_image(image_id, False)
        
        axes[r, 0].imshow(image)
        axes[r, 0].set_title(f'Image {image_id} Raw', size=16)
        axes[r, 0].axis(False)

        mask = rle2mask(image_id)
        axes[r, 1].imshow(mask)
        axes[r, 1].set_title('Mask', size=16)
        axes[r, 1].axis(False)
        
        axes[r, 2].imshow(image)
        axes[r, 2].imshow((mask * np.array([255, 0, 0])), alpha=0.50)
        axes[r, 2].set_title('Image and Mask', size=16)
        axes[r, 2].axis(False)
            
    fig.subplots_adjust(wspace=0.10)
    plt.show()

def maskexport():
	image_ids = train['id']
	for image_id in image_ids:
		mask = rle2mask(image_id)
		img = plt.imshow(mask)
		plt.axis(False)
		plt.savefig(f'E:/machine_learning/HumanHacking_project/masks/{image_id}.jpg', bbox_inches='tight', pad_inches=0)
		

def specialexport():
	img_ids = [1229, 2344, 2668, 5102, 5317, 6794, 9450, 9769, 11645,
	12784, 14396, 14756, 15860, 16149, 17455, 18121, 18777, 21112,
	23760, 28318, 28622, 28657, 28748, 28963, 29296]
	for img_id in img_ids:
		mask_1 = rle2mask(train[train["id"]==img_id]["rle"].iloc[-1], 
			(train[train["id"]==img_id]["img_height"].iloc[-1], train[train["id"]==img_id]["img_width"].iloc[-1]))
		plt.imshow(mask_1)
		plt.axis(False)
		plt.savefig(f'E:/machine_learning/HumanHacking_project/masks/{img_id}.jpg', bbox_inches='tight', pad_inches=0)
