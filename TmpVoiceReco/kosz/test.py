from numpy import *
from numpy.linalg import *
from scipy.fftpack import dct
from scipy.io import wavfile
import RecordModule
import pylab
import numpy

def melFilterBank(blockSize):
    numBands = int(numCoefficients)
    maxMel = int(freqToMel(maxHz))
    minMel = int(freqToMel(minHz))

    # Create a matrix for triangular filters, one row per filter
    filterMatrix = zeros((numBands, blockSize))

    melRange = array(range(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    # each array index represent the center of each triangular filter
    aux = log(1 + 1000.0 / 700.0) / 1000.0
    aux = (exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = floor(aux)  # Arredonda pra baixo
    centerIndex = array(aux, int)  # Get int values

    for i in range(numBands):
        start, centre, end = centerIndex[i:i + 3]
        k1 = float32(centre - start)
        k2 = float32(end - centre)
        up = (array(range(start, centre)) - start) / k1
        down = (end - array(range(centre, end))) / k2

        filterMatrix[i][start:centre] = up
        filterMatrix[i][centre:end] = down

    return filterMatrix.transpose()

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048 - 1))









for i in range(10):
    filename = "learn_set//wylacz//"+str(i+1)+".wav"
    
#############
    sampleRate, signal = wavfile.read(filename)
    numCoefficients = 16 # choose the sice of mfcc array
    minHz = 0
    maxHz = 8000
    
    complexSpectrum = fft.fft(signal)
    powerSpectrum = abs(complexSpectrum) ** 2
    filteredSpectrum = dot(powerSpectrum, melFilterBank(len(powerSpectrum)))
    logSpectrum = numpy.log(filteredSpectrum)
    dctSpectrum = dct(logSpectrum, type=2)  # MFCC :)
#############
    ceps = dctSpectrum.T
    vect_of_mccf = ceps
    
    pylab.title("ceps : ") 
    pylab.plot(range(len(vect_of_mccf)), vect_of_mccf, 'g')
pylab.show()