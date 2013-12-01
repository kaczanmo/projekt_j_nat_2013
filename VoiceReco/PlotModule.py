'''
Created on 23-10-2013

@author: Tenac
'''


import scipy
import scipy.fftpack
import pylab
from scipy import pi
import wave
import numpy as np
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange
from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
import numpy
import RecordModule


def plotSpectrum(y,Fs):
 """
 Plots a Single-Sided Amplitude Spectrum of y(t)
 """
 maxFreq = 5000
 n = len(y) # length of the signal
 k = arange(n)
 T = n/Fs
 frq = k/T # two sides frequency range

#  frq = frq[0:n/2] # one side frequency range
 frq = frq[0:(maxFreq)]
 print(frq.shape)
 Y = fft(y)/n # fft computing and normalization
#  Y = Y[0:n/2]
 Y = Y[0:(maxFreq)]
 
 plot(frq,abs(Y),'r') # plotting the spectrum
 xlabel('Freq (Hz)')
 ylabel('|Y(freq)|')


def readWav(filename, Fs):
 rate,data=read(filename)
 y = numpy.atleast_2d(data)[0]
 lungime=len(y)
 timp=len(y)/Fs
 t=linspace(0,timp,len(y))
 return t, y


if __name__ == '__main__':
    Fs = 44100.0;  # sampling rate
    filename = "learn_set//wlacz//6.wav"
    t,y = readWav(filename, Fs)
    y = RecordModule.preemp(y)
    fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y = RecordModule.detectSingleWord(t,y)

    pylab.subplot(211)
    pylab.title(filename) 
    pylab.plot(t, y)
    pylab.subplot(212)
    print(word_y.shape)
    plotSpectrum(word_y,Fs)
    pylab.show()