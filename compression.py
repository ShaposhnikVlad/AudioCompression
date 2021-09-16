import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
import matplotlib.pyplot as plt

from pathlib import Path

from common import split_to_blocks


def normalize(data: np.ndarray) -> np.ndarray:
    """ Normalize input array

    :param data: array to normalize
    :return: Array in +-1 range
    """
    sign = 1 if np.max(data) == np.max(np.abs(data)) else -1
    return data / (np.max(np.abs(data)) * sign)


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


def compression_stage_1(wave: np.ndarray, block: int, export_name: str, show_polts: bool = False) -> np.ndarray:
    """Perform first stage of file compression """
    N_points = 100000
    n_bins = 1000
    norm_wave = normalize(wave)

    if show_polts:
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        fig.suptitle("Normalization", fontsize="x-large")
        axs[0].hist(wave, bins=n_bins)
        axs[0].set_title("Original")
        axs[0].set_ylabel('Count')
        axs[0].set_xlabel('Power')
        axs[1].hist(norm_wave, bins=n_bins)
        axs[1].set_title("Normalized")
        fig.show()
        fig.savefig(f'results/{export_name}_normalization.png')

    blocks = split_to_blocks(block, norm_wave)
    print(f"Source array shape: {norm_wave.shape}, matrix shape: {blocks.shape}")

    if show_polts:
        plt.imshow(blocks)
        plt.colorbar()
        plt.xlabel("Sample")
        plt.ylabel("Blocks")
        plt.title("Audio blocks")
        plt.show()
        plt.imsave(f'results/{export_name}_blocks.png', blocks)

    return blocks


def compression_stage_2(blocks: np.ndarray, export_name: str, show_polts: bool = False) -> np.ndarray:
    coeff = calculate_dct(blocks)

    if show_polts:
        image = plt.imshow(coeff)
        plt.colorbar()
        plt.xlabel("Coefficient")
        plt.ylabel("Blocks")
        plt.title("DCT Matrix")
        plt.show()
        plt.savefig(f'results/{export_name}_DCT.png')
        plt.plot(coeff[0], label="Block 1")
        plt.plot(coeff[1], label="Block 2")
        plt.plot(coeff[-1], label="Block N-1")
        plt.xlabel("Sample")
        plt.ylabel("Energy")
        plt.title("DCT coefficients")
        plt.legend()
        plt.show()
        plt.imsave(f'results/{export_name}_DCT2.png', blocks)

    return coeff


def compression_stage_3(coeff: np.ndarray, export_name: str, show_polts: bool = False) -> list:
    q_coef = quantize(data=coeff)
    if show_polts:
        plt.plot(q_coef[0], label="Block 1")
        plt.plot(q_coef[1], label="Block 2")
        plt.plot(q_coef[-1], label="Block N-1")
        plt.xlabel("Sample")
        plt.ylabel("Energy")
        plt.title("Quantized DCT coefficients")
        plt.legend()
        plt.show()
        plt.savefig(f'results/{export_name}_Q_DCT.png')
        plt.imshow(q_coef)
        plt.colorbar()
        plt.xlabel("Coefficient")
        plt.ylabel("Blocks")
        plt.title("Quantized DCT Matrix")
        plt.show()
        plt.imsave(f'results/{export_name}_Q_DCT_BLOCKS.png', q_coef)

    chars_data = [chr(c) for c in np.asarray(q_coef).reshape(-1).tolist()]

    return chars_data


def compression_stage_4(chars_data: list) -> list:
    compress_data = compress(chars_data)
    print(f"Uncompressed length: {len(chars_data)}")
    print(f"Compressed length: {len(compress_data)}")
    print(f"Data compression ratio: {len(chars_data) / len(compress_data)}")
    return compress_data
