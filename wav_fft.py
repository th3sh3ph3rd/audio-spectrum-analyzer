#!/usr/bin/python

import sys
import numpy as np
from scipy.io.wavfile import read

NBANDS = 16 # number of bins for the final FFT output
SAMP_RATE = 44100
FFT_RES = 10 # resolution in Hz per FFT value, should be power of 10
NFFT = int(SAMP_RATE/FFT_RES)

wavFile = read("foo.wav")
audio = np.array(wavFile[1], dtype=np.float64) # take one of the two stereo channels
audio = audio[:int(audio.shape[0]/NFFT)*NFFT] # clip audio data to match FFT size
audio = np.reshape(audio, (-1,NFFT))

# compute row-wise FFT
spectrum = np.absolute(np.fft.fft(audio, axis=1))/NFFT

# get the band boundaries in Hz, matched to the FFT resolution
fftBandBounds = np.around(np.logspace(np.log10(20), np.log10(20000), NBANDS)/FFT_RES, decimals=0)*FFT_RES
print(fftBandBounds)

spectrumBinned = np.zeros([spectrum.shape[0], NBANDS])

# average the spectrum value into the logarithmic bands 
for rowIdx in range(spectrum.shape[0]):
    freq = 0
    fft_cnt = 0
    bandIdx = 0
    for fftIdx in range(NFFT):
        spectrumBinned[rowIdx, bandIdx] += spectrum[rowIdx, fftIdx]
        freq += FFT_RES
        fft_cnt += 1
        if freq > fftBandBounds[bandIdx]:
            spectrumBinned[rowIdx, bandIdx] /= fft_cnt
            fft_cnt = 0
            bandIdx += 1
            if bandIdx == NBANDS:
                break

# convert to single precision float for js compatibility
spectrumBinned = spectrumBinned.astype(np.single)

# write data to file
f = open("foo.txt", "w")
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float_kind':"{:1f}".format})
f.write(np.array2string(spectrumBinned, max_line_width=sys.maxsize, separator=',', threshold=sys.maxsize)[1:-1])
f.close()

