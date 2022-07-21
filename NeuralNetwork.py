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
    return cv2.resize(img, [img_size, img_size], interpolation=cv2.INTER_CUBIC).astype(np.uint8)


def rle2mask(image_id, shape=(img_size, img_size)):
	row = train.loc[train['id'] == image_id].squeeze()

	s = row['rle'].split()

	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths

	mask = np.zeros(shape=[shape[0] * shape[1]], dtype=np.uint8)

	for lo, hi in zip(starts, ends):
		mask[lo : hi] = 1

	mask = mask.reshape(shape).T
	mask = resize_img(mask)
	mask = np.expand_dims(mask, axis=2)

	return mask

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

'''
image_gen = ImageDataGenerator(rescale=1/255)

train = image_gen.flow_from_directory(
		train_path,
		target_size=(img_height, img_width),
		color_mode='grayscale',
		batch_size=64,
	)

test = test_data_gen.flow_from_directory(
		mask_path,
		target_size=(img_height, img_width),
		color_mode='grayscale',
		batch_size=1
	)
'''

def maskexport():
	image_ids = train['id']
	for image_id in image_ids:
		mask = rle2mask(image_id)
		img = plt.imshow(mask)
		plt.axis(False)
		plt.savefig(f'E:/machine_learning/HumanHacking_project/masks/{image_id}.jpg', bbox_inches='tight', pad_inches=0)


mask = rle2mask(1229)
img = plt.imshow(mask)
plt.axis(False)
plt.savefig(f'E:/machine_learning/HumanHacking_project/{image_id}.jpg', bbox_inches='tight', pad_inches=0)