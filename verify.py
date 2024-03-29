#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dirname')
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import color, io
import math
import os

seg_filenames = [os.path.join(args.dirname, f) for f in sorted(os.listdir(args.dirname)) if f.endswith('_seg.png')]

def key_pressed(ev):
    if ev.key in ('enter', 'escape'):
        plt.close()

def mask_bbox(mask, border=10):
    a = np.where(mask)
    return (np.maximum(np.min(a[0])-border, 0),
        np.minimum(np.max(a[0])+border, mask.shape[0]),
        np.maximum(np.min(a[1])-border, 0),
        np.minimum(np.max(a[1])+border, mask.shape[1]))

size = 2
end = math.ceil(len(seg_filenames)/size)*size
for i in range(0, end, size):
    for j, k in enumerate(range(i, i+size)):
        seg_filename = seg_filenames[k]
        img_filename = seg_filename[:-8] + '.png'
        mask = io.imread(seg_filename, True)
        img = io.imread(img_filename, True)

        img = color.gray2rgb(img)
        bbox = mask_bbox(mask)
        img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        plt.subplot(1, 2, j+1)
        plt.imshow(img)
        if len(contours) > 1:
            plt.text(img.shape[1]/2, img.shape[0]/2, f'Weird: there are {len(contours)} contours!', ha='center', va='center', color='red')
        plt.title(os.path.basename(img_filename))
    plt.suptitle(f'{i+1}-{i+size} / {end}')
    plt.get_current_fig_manager().window.showMaximized()
    plt.gcf().canvas.mpl_connect('key_press_event', key_pressed)
    plt.show()
