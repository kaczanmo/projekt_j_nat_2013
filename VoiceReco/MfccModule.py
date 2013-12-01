'''
Created on 25-10-2013

@author: Tenac
'''
import numpy
import PlotModule
from scipy.fftpack import dct
from scipy.io import wavfile
import sys
import math
from scipy import fft, arange, ifft
import scipy
import scipy.fftpack
import pylab


Fs = 44100.0;  # sampling rate
numCoefficients = 16 # choose the sive of mfcc array
minHz = 200
maxHz = 1000 
    

def trfbank(fs=16000, nfft=512, lowfreq = 133.33, linsc = 200/3. , logsc = 1.0711703,  nlinfilt = 13 , nlogfilt = 27):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = numpy.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** numpy.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]
#         print ("i:",low, " c:",cen, " h:", hi)

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1,
                        numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                        numpy.floor(hi * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (hi - cen)

        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])
#         print (fbank)
    return fbank, freqs

def melFilterBank(blockSize):
    numBands = int(numCoefficients)
    maxMel = int(freqToMel(maxHz))
    minMel = int(freqToMel(minHz))
    
#     print (maxMel, "  ", minMel)

    # Create a matrix for triangular filters, one row per filter
    filterMatrix = numpy.zeros((numBands, blockSize))

    melRange = numpy.array(range(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    # each array index represent the center of each triangular filter
    aux = numpy.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (numpy.exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = numpy.floor(aux)  # Arredonda pra baixo
    centerIndex = numpy.array(aux, int)  # Get int values

    for i in range(numBands):
        start, centre, end = centerIndex[i:i + 3]
        k1 = numpy.float32(centre - start)
        k2 = numpy.float32(end - centre)
        up = (numpy.array(range(start, centre)) - start) / k1
        down = (end - numpy.array(range(centre, end))) / k2

        filterMatrix[i][start:centre] = up
        filterMatrix[i][centre:end] = down
#     print ("ff:",filterMatrix.shape)
    return filterMatrix.transpose()

def freqToMel(freq):
    return 1127.01048 * math.log((freq / 700.0) + 1)

def melToFreq(mel):
    return 700 * (math.exp((mel / 1127.01048) - 1))

def getCepsVect(signal):
    complexSpectrum = fft(signal)
    powerSpectrum = abs(complexSpectrum) **2
    
    filteredSpectrum = numpy.dot(powerSpectrum, melFilterBank(len(powerSpectrum)) )
    logSpectrum = numpy.log(filteredSpectrum)
    dctSpectrum = dct(logSpectrum, type=2)  # MFCC :)
    dctSpectrum[0] = 0 
    
    return dctSpectrum
    
    
    
    
if __name__ == '__main__':

    for i in range(10,11):
        filename = "learn_set//wlacz//"+str(i+1)+".wav"
        sampleRate, signal = PlotModule.readWav(filename, Fs)
        ceps = getCepsVect(signal)
        pylab.title(filename) 
        pylab.plot(range(numCoefficients), ceps, 'b')
        
#     for i in range(11):
#         filename = "learn_set//wylacz//"+str(i+1)+".wav"
#         sampleRate, signal = PlotModule.readWav(filename, Fs)
#         ceps = getCepsVect(signal)
#         pylab.title(filename) 
#         pylab.plot(range(numCoefficients), ceps, 'r')
        
        
    pylab.show()