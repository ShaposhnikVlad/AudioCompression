from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct, idct


def normalize(data: np.ndarray) -> np.ndarray:
    """ Normilize input array

    :param data: array to normalize
    :return: Array in +-1 range
    """
    sign = 1 if np.max(data) == np.max(np.abs(data)) else -1
    return data / (np.max(np.abs(data)) * sign)


def split_to_blocks(block_size: int, data: np.ndarray) -> np.ndarray:
    """ Split 1D array into 2D matrix

    :param block_size: Block size
    :param data: Input data
    :return: Spliced data
    """
    return np.reshape(data, (-1, block_size))


def calculate_dct(data: np.ndarray) -> np.ndarray:
    """ Calculate DCT over input matrix

    :param data: Input 2D matrix
    :return: DCT coefficients
    """

    def _dct(d):
        return dct(d, type=2, norm='ortho')

    return np.apply_along_axis(_dct, axis=0, arr=data)


def quantize(data: np.ndarray) -> np.ndarray:
    """ Quantize input array in 255 levels

    :param data: Input matrix
    :return: Quantized matrix
    """

    def _quantize(d):
        return np.digitize(d, np.arange(0, 255))

    return np.apply_along_axis(_quantize, axis=0, arr=(data + 1) * 128)


def compress(uncompressed: list) -> list:
    """Compress a string to a list of output symbols.

    :param uncompressed: Input array
    :return: Compressed array
    """

    # Build the dictionary.
    dict_size = 256
    dictionary = {chr(i): chr(i) for i in range(dict_size)}

    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result


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


if __name__ == '__main__':
    N_points = 100000
    n_bins = 1000
    block = 1024
    wave = np.array(wavfile.read("data/metal/metal.00000.wav")[1], dtype=float)
    norm_wave = normalize(wave)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.suptitle("Normalization", fontsize="x-large")

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(wave, bins=n_bins)
    axs[0].set_title("Original")
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('Power')
    axs[1].hist(norm_wave, bins=n_bins)
    axs[1].set_title("Normalized")
    blocks = split_to_blocks(block, norm_wave)
    print(f"Source array shape: {norm_wave.shape}, matrix shape: {blocks.shape}")
    fig.show()
    fig.savefig('results/normalization.png')
    plt.imshow(blocks)
    plt.colorbar()
    plt.xlabel("Sample")
    plt.ylabel("Blocks")
    plt.title("Audio blocks")
    plt.show()
    plt.savefig('results/blocks.png')
    coeff = calculate_dct(blocks)
    plt.imshow(coeff)
    plt.colorbar()
    plt.xlabel("Coefficient")
    plt.ylabel("Blocks")
    plt.title("DCT Matrix")
    plt.show()
    plt.savefig('results/DCT.png')
    plt.plot(coeff[0], label="Block 1")
    plt.plot(coeff[1], label="Block 2")
    plt.plot(coeff[-1], label="Block N-1")
    plt.xlabel("Sample")
    plt.ylabel("Energy")
    plt.title("DCT coefficients")
    plt.legend()
    plt.show()
    plt.savefig('results/DCT2.png')
    q_coef = quantize(data=coeff)
    plt.plot(q_coef[0], label="Block 1")
    plt.plot(q_coef[1], label="Block 2")
    plt.plot(q_coef[-1], label="Block N-1")
    plt.xlabel("Sample")
    plt.ylabel("Energy")
    plt.title("Quantized DCT coefficients")
    plt.legend()
    plt.show()
    plt.savefig('results/Q_DCT.png')
    plt.imshow(q_coef)
    plt.colorbar()
    plt.xlabel("Coefficient")
    plt.ylabel("Blocks")
    plt.title("Quantized DCT Matrix")
    plt.show()
    plt.savefig('results/Q_DCT_BLOCKS.png')

    chars_data = [chr(c) for c in np.asarray(q_coef).reshape(-1).tolist()]

    compress_data = compress(chars_data)
    print(f"Uncompressed length: {len(chars_data)}")
    print(f"Compressed length: {len(compress_data)}")
    print(f"Data compression ratio: {len(chars_data) / len(compress_data)}")


