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
import functools
import itertools
import math

import numpy as np
import cv2 as cv

import utils


def main():
    # parse the command line arguments to attributes of 'args'
    parser = argparse.ArgumentParser(description='DCT compressor. '
                                                 'Reconstructs an image using only the first K coefficients '
                                                 'of its 8x8 DCT.')
    parser.add_argument('--input', dest='image_path', required=True, type=str,
                        help='Path to the image to be analyzed.')
    parser.add_argument('--coeffs', dest='num_coeffs', required=True,
                        type=functools.partial(utils.power_of_two, max=64),
                        help='Number of coefficients that will be used to reconstruct the original image. 64 max.')
    parser.add_argument('--blocksize', dest='blocksize', type=int, default=8,
                        help='Length of the sides of the blocks the input image will be cut into.')
    args = parser.parse_args()

    # read image
    img = cv.imread(args.image_path, cv.CV_LOAD_IMAGE_COLOR)
    img = np.float32(img) / 255

    # get YCC components
    img_ycc = cv.cvtColor(img, code=cv.cv.CV_BGR2YCrCb, dstCn=3)

    # compress and decompress every channel separately
    rec_img = np.empty_like(img)
    for channel_num in xrange(3):
        mono_image = approximate_mono_image(img_ycc[:, :, channel_num],
                                            num_coeffs=args.num_coeffs,
                                            blocksize=args.blocksize)
        rec_img[:, :, channel_num] = mono_image

    # round to the nearest integer [0,255] value
    rec_img = np.uint8(np.round(rec_img * 255))

    # convert back to RGB
    rec_img_rgb = cv.cvtColor(rec_img, code=cv.cv.CV_YCrCb2BGR, dstCn=3)

    # visualize
    cv.imshow('', rec_img_rgb)
    while True:
        cv.waitKey(33)


def approximate_mono_image(img, num_coeffs, blocksize=8):
    """
    Approximates a single channel image by using only the first coefficients of the DCT.
     First, the image is chopped into squared patches. Then the DCT is applied to each patch and only the K coefficients
     that correspond to the upper left corner of the 2D DCT basis are kept.
     Finally, only these coefficients are used to approximate the original patches with the IDCT, and the image is
     reconstructed back again from these patches.
    :param img: Image to be approximated.
    :param num_coeffs: Number of DCT coefficients to use (the upper left corner).
    :param blocksize: Length of the sides of the blocks the input image will be cut into.
    :return: The approximated image.
    """

    # prevent against multiple-channel images
    if len(img.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')

    # shape of image
    height = img.shape[0]
    width = img.shape[1]
    if (height % blocksize != 0) or (width % blocksize != 0):
        raise ValueError("Image dimensions are not multiple of the blocksize")

    # split into blocksize x blocksize pixels blocks
    img_blocks = [img[j:j + blocksize, i:i + blocksize]
                  for (j, i) in itertools.product(range(0, height, blocksize),
                                                  range(0, width, blocksize))]

    # DCT transform every 8x8 block
    dct_blocks = [cv.dct(img_block) for img_block in img_blocks]

    # keep only the upper left KxK corner of the DCT coefficients of every block
    reduced_dct_coeffs = []
    for dct_block in dct_blocks:
        some_dct_coeffs = np.zeros_like(dct_block)
        logg = math.log(num_coeffs, 2)
        some_dct_coeffs[0:logg, 0:logg] = dct_block[0:logg, 0:logg]
        reduced_dct_coeffs.append(some_dct_coeffs)

    # IDCT of every block
    rec_img_blocks = [cv.idct(coeff_block) for coeff_block in reduced_dct_coeffs]

    # reshape the reconstructed image blocks
    rec_img = []
    for chunk_row_blocks in utils.chunks(rec_img_blocks, width / blocksize):
        for row_block_num in xrange(blocksize):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])
    rec_img = np.array(rec_img).reshape(height, width)

    return rec_img


if __name__ == '__main__':
    main()