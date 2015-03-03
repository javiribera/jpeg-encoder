import argparse
import numpy as np
import math

def power_of_two(number, max=np.inf):
    """
    Convert a number to a power of two, below a given maximum number.
    :param number: The number to be converted to a power of two.
    :param max: Maximum number the input number should be.
    :return: The positive integer.
    :raise: argparse.ArgumentTypeError if not able to do the conversion.
    """
    try:
        number_float = float(number)

        if int(math.log(number_float, 2)) == math.log(number_float, 2) \
                and 0 < number_float <= max:
            return int(number_float)
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('%s is not a power of two <= %s' % (number, max))


def chunks(l, n):
    """ Yield successive n-sized chunks from l """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]