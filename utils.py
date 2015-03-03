import argparse
import numpy as np
import math

def perfect_square(number, max=np.inf):
    """
    Convert a number to a perfect square, below a given maximum number, if possible.
    :param number: The number to be converted to a perfect square.
    :param max: Maximum number the input number should be.
    :return: The positive integer.
    :raise: argparse.ArgumentTypeError if not able to do the conversion.
    """
    try:
        number_float = float(number)

        if int(math.sqrt(number_float)) == math.sqrt(number_float) \
                and 0 < number_float <= max:
            return int(number_float)
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('%s is not a perfect square <= %s' % (number, max))


def chunks(l, n):
    """ Yield successive n-sized chunks from l """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]