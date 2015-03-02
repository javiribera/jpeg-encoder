import math

__author__ = 'javiribera'

"""
DCT compressor.
Reconstructs an image using only the first K coefficients of its 8x8 DCT.

Tried in Python 2.7.5

!! DEPENDENCIES !!
This script depends on the following external Python Packages:
- argparse, to parse command line arguments
- OpenCV, for the DCT and I/O of images
- numpy, to speed up array manipulation
"""

import argparse
import sys
import functools
import itertools

# from ipdb import set_trace as debug
import numpy as np

import utils

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2 as cv


# parse the command line arguments to attributes of 'args'
parser = argparse.ArgumentParser(description='DCT compressor. '
                                             'Reconstructs an image using only the first K coefficients '
                                             'of its 8x8 DCT.')
parser.add_argument('--input', dest='image_path', required=True, type=str,
                    help='Path to the image to be analyzed.')
parser.add_argument('--coeffs', dest='num_coeffs', required=True,
                    type=functools.partial(utils.power_of_two, max=64),
                    help='Number of coefficients that will be used to reconstruct the original image. 64 max.')
args = parser.parse_args()

# read image
img = cv.imread(args.image_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
img = np.float64(img)

# shape of image
height = img.shape[0]
width = img.shape[1]
if (height % 8 != 0) or (width % 8 != 0):
    raise ValueError("Image '%s' dimensions are not multiple of 8 pixels")

# split in 8x8 pixels blocks
img_blocks = [img[j:j + 8, i:i + 8] for (j, i) in itertools.product(range(0, height, 8),
                                                                    range(0, width, 8))]

# DCT transform every 8x8 block
dct_blocks = [cv.dct(img_block) for img_block in img_blocks]

# keep only the upper left KxK corner of the DCT coefficients of every block
reduced_dct_coeffs = []
for dct_block in dct_blocks:
    some_dct_coeffs = np.zeros_like(dct_block)
    logg = math.log(args.num_coeffs, 2)
    some_dct_coeffs[0:logg, 0:logg] = dct_block[0:logg, 0:logg]
    reduced_dct_coeffs.append(some_dct_coeffs)

# IDCT of every block
rec_img_blocks = [cv.idct(coeff_block) for coeff_block in reduced_dct_coeffs]

# reshape the reconstructed image blocks
rec_img = []
for chunk_row_blocks in utils.chunks(rec_img_blocks, width / 8):
    for row_block_num in xrange(8):
        for block in chunk_row_blocks:
            rec_img.extend(block[row_block_num])
rec_img = np.array(rec_img).reshape(512, 512)

# round to the nearest integer [0,255] value
rec_img = np.uint8(np.round(rec_img))

cv.imshow('', np.uint8(rec_img))
while True:
    cv.waitKey(33)

# debug()
