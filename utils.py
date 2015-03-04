import argparse
import numpy as np

def chunks(l, n):
    """ Yield successive n-sized chunks from l """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def zig_zag(array, n=None):
    """
    Return a new array where only the first n subelements in zig-zag order are kept.
    The remaining elements are set to 0.
    :param array: 2D array_like
    :param n: Keep up to n subelements. Default: all subelements
    :return: The new reduced array.
    """

    shape = np.array(array).shape

    assert len(shape) >= 2, "Array must be a 2D array_like"

    if n==None:
        n = shape[0]*shape[1]
    assert 0 <= n <= shape[0]*shape[1], 'n must be the number of subelements to return'

    res = np.zeros_like(array)

    (j, i) = (0, 0)
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    for subel_num in xrange(1,n+1):
        res[j][i] = array[j][i]
        if direction == 'r':
            i += 1
            if j == shape[0] - 1:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            i -= 1
            j += 1
            if j == shape[0] - 1:
                direction = 'r'
            elif i == 0:
                direction = 'd'
        elif direction == 'd':
            j += 1
            if i == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            i += 1
            j -= 1
            if i == shape[1] - 1:
                direction = 'd'
            elif j == 0:
                direction = 'r'

    return res

