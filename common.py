import numpy as np


def split_to_blocks(block_size: int, data: np.ndarray) -> np.ndarray:
    """ Split 1D array into 2D matrix

    :param block_size: Block size
    :param data: Input data
    :return: Spliced data
    """
    data_len = (data.shape[0] // block_size) * block_size
    return np.reshape(data[0: data_len], (-1, block_size))