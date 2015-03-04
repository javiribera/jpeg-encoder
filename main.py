"""
DCT compressor.
Reconstructs an image using only the first K coefficients of its 8x8 DCT.

Tried in Python 2.7.5

!! DEPENDENCIES !!
This script depends on the following external Python Packages:
- argparse, to parse command line arguments
- OpenCV, for the DCT and I/O of images
- numpy, to speed up array manipulation
- scikit-image, for SSIM computation
"""

import argparse
import itertools
import math
from skimage.measure import structural_similarity as compute_ssim

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
    parser.add_argument('--coeffs', dest='num_coeffs', required=True, type=int,
                        help='Number of coefficients that will be used to reconstruct the original image.')
    args = parser.parse_args()

    # read image
    orig_img = cv.imread(args.image_path, cv.CV_LOAD_IMAGE_COLOR)
    img = np.float32(orig_img)

    # get YCC components
    img_ycc = cv.cvtColor(img, code=cv.cv.CV_BGR2YCrCb, dstCn=3)

    # compress and decompress every channel separately
    rec_img = np.empty_like(img)
    for channel_num in xrange(3):
        mono_image = approximate_mono_image(img_ycc[:, :, channel_num],
                                            num_coeffs=args.num_coeffs)
        rec_img[:, :, channel_num] = mono_image

    # convert back to RGB
    rec_img_rgb = cv.cvtColor(rec_img, code=cv.cv.CV_YCrCb2BGR, dstCn=3)

    # round to the nearest integer [0,255] value
    rec_img_rgb = np.uint8(np.round(rec_img_rgb))

    # show PSNR and SSIM of the approximation
    err_img = abs(np.array(rec_img_rgb, dtype=float) - np.array(orig_img, dtype=float))
    mse = (err_img ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    ssim = compute_ssim(cv.cvtColor(np.float32(rec_img_rgb), code=cv.cv.CV_BGR2GRAY),
                        cv.cvtColor(np.float32(orig_img), code=cv.cv.CV_BGR2GRAY))
    print 'PSNR: %s dB' % psnr
    print 'SSIM: %s' % ssim

    # visualize
    cv.imshow('Approximation image', rec_img_rgb)
    while True:
        cv.waitKey(33)


def approximate_mono_image(img, num_coeffs):
    """
    Approximates a single channel image by using only the first coefficients of the DCT.
     First, the image is chopped into 8x8 pixels patches.
     Then the DCT is applied to each patch and only the first K DCT coefficients are kept.
     Finally, only these coefficients are used to approximate the original patches with the IDCT, and the image is
     reconstructed back again from these patches.
    :param img: Image to be approximated.
    :param num_coeffs: Number of DCT coefficients to use.
    :return: The approximated image.
    """

    # prevent against multiple-channel images
    if len(img.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')

    # shape of image
    height = img.shape[0]
    width = img.shape[1]
    if (height % 8 != 0) or (width % 8 != 0):
        raise ValueError("Image dimensions (%s, %s) must be multiple of 8" %(height, width))

    # split into 8 x 8 pixels blocks
    img_blocks = [img[j:j + 8, i:i + 8]
                  for (j, i) in itertools.product(xrange(0, height, 8),
                                                  xrange(0, width, 8))]

    # DCT transform every 8x8 block
    dct_blocks = [cv.dct(img_block) for img_block in img_blocks]

    # keep only the first K DCT coefficients of every block
    reduced_dct_coeffs = [utils.zig_zag(dct_block, num_coeffs) for dct_block in dct_blocks]

    # IDCT of every block
    rec_img_blocks = [cv.idct(coeff_block) for coeff_block in reduced_dct_coeffs]

    # reshape the reconstructed image blocks
    rec_img = []
    for chunk_row_blocks in utils.chunks(rec_img_blocks, width / 8):
        for row_block_num in xrange(8):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])
    rec_img = np.array(rec_img).reshape(height, width)

    return rec_img


if __name__ == '__main__':
    main()