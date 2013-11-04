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
from audiolazy import sHz, sin_table, str2freq, lpc



Fs = 44100.0;  # sampling rate
numCoefficients = 13 # choose the sive of mfcc array
minHz = 0
maxHz = 8000 
    


def getCooefVect(signal):
#     cooef  = [1]*numCoefficients
    len(signal) # Small data
    filt = lazy_lpc.lpc.kautocor(signal, 2)
    filt # The analysis filter
    cooef = filt.numerator # List of coefficients

    filt.error # Prediction error (squared!)

    return cooef
    
    
    
    
if __name__ == '__main__':
    
    i = 0
    filename = "learn_set//wlacz//"+str(i+1)+".wav"
    sampleRate, signal = PlotModule.readWav(filename, Fs)
    
    rate = 22050
    s, Hz = sHz(rate)
    size = 512
    table = signal  #sin_table.harmonize({1: 1, 2: 5, 3: 3, 4: 2, 6: 9, 8: 1}).normalize()
    
    data = table(str2freq("Bb3") * Hz).take(size)
    filt = lpc(signal, order=14) # Analysis filter
    gain = 1e-2 # Gain just for alignment with DFT
    
    # Plots the synthesis filter
    # - If blk is given, plots the block DFT together with the filter
    # - If rate is given, shows the frequency range in Hz
    (gain / filt).plot(blk=data, rate=rate, samples=1024, unwrap=False)
    pylab.show()

#     for i in range(10):
#         filename = "learn_set//wlacz//"+str(i+1)+".wav"
#     
#         sampleRate, signal = PlotModule.readWav(filename, Fs)
#       
#         
#         cooef = getCooefVect(signal)
#         
#         
#         pylab.title(filename) 
#         pylab.plot(range(len(cooef)), cooef, 'b')
#     pylab.show()