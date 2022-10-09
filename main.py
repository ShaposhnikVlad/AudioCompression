# Shaposhnik Vladislav 09.10.22
from __future__ import annotations
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

from compression import compression_stage_1, compression_stage_2, compression_stage_3, compression_stage_4
from decompression import decompress_stage_1, decompress_stage_2, decompress_stage_3

from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wavfile.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return ims

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


if __name__ == '__main__':
    # Config
    block = 1024
    # Source
    file_path = Path("data/country/country.00004.wav")
    export = "metal"
    print(f"Processing file: {file_path}")
    rate, wave = wavfile.read(file_path)
    wave = np.array(wave, dtype=float)
    # Compression
    blocks = compression_stage_1(wave, block, export, False)
    coeff = compression_stage_2(blocks, export, False)
    chars = compression_stage_3(coeff, export, False)
    compressed_data = compression_stage_4(chars)
    #  Reconstruction
    d_blocks = decompress_stage_1(compressed_data)
    d_coef = decompress_stage_2(d_blocks, block)
    r_wave = decompress_stage_3(d_blocks)
    wavfile.write("test.wav", rate, r_wave)
    # Analysis
    o_wave = np.asarray(blocks).reshape(-1)  # Convert normalized block into stream
    mse = (np.square(o_wave - r_wave)).mean(axis=0)
    print(f"Audio MSE : {mse}")
    s_snr = 10 * np.log10(signaltonoise(wave))
    d_snr = 10 * np.log10(signaltonoise(r_wave))
    print(f"Source SNR: {s_snr}, Decompressed SNR: {d_snr}")
    d_range = 20 * np.log10(np.abs(wave.max()/wave.min()))
    psnr = 10 * np.log10((d_range**2) / mse)
    print(f"PSNR: {psnr}, dynamic range: {d_range}")