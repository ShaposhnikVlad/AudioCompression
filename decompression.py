import numpy as np
from scipy.fftpack import idct

import matplotlib.pyplot as plt

from common import split_to_blocks


def decompress(compressed: list) -> list:
    """Decompress a list of output ks to a string.

    :param compressed: Compressed string
    :return: Recovered array
    """

    # Build the dictionary.
    dict_size = 256
    dictionary = {chr(i): chr(i) for i in range(dict_size)}

    w = result = compressed.pop(0)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result += entry

        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry
    return result


def calculate_inverse_dct(data: np.ndarray) -> np.ndarray:
    """ Calculate iDCT over input matrix

    :param data: DCT coefficients
    :return: 2D matrix
    """

    def _idct(d):
        return idct(d, type=2, norm='ortho')

    return np.apply_along_axis(_idct, axis=0, arr=data)


def decompress_stage_1(chars: list) -> np.ndarray:
    """Reconstruct DCT coefficients from compressed blocks

    :param chars: List of compressed DCT coefficients
    :return uncompressed DCT coefficients
    """
    d = [ord(c) for c in decompress(chars)]
    return np.array(d)


def decompress_stage_2(coef: np.ndarray, block) -> np.ndarray:
    """ Reconstruct audio blocks from DCT coefficients

    :param coef: DCT coefficients
    :param block: block size
    :return: Reconstructed audio blocks
    """
    blocks = split_to_blocks(block_size=block, data=coef)
    return calculate_inverse_dct(blocks)


def decompress_stage_3(blocks: np.ndarray) -> np.ndarray:
    """Convert audio blocks into stream

    :param blocks: Audio blocks
    :return Audio stream
    """
    return np.squeeze(np.asarray(blocks))