'''
Created on 25-10-2013

@author: Tenac
'''

from scipy.io import loadmat
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import PlotModule
import numpy as np
import warnings
import pylab

from numpy import meshgrid, dot
from numpy.linalg import inv
from numpy import sqrt, cos, pi
from matplotlib.pyplot import *
from numpy.core.numeric import zeros
import RecordModule
import numpy





def dctmtx(n):
    """
    Return the DCT-II matrix of order n as a numpy array.
    """
    x,y = meshgrid(range(n), range(n))
    D = sqrt(2.0/n) * cos(pi * (2*x+1) * y / (2*n))
    D[0] /= sqrt(2)
    return D

FS = 44100.0                              # Sampling rate
FRAME_LEN = int(0.02 * FS)              # Frame length
FRAME_SHIFT = int(0.01 * FS)            # Frame shift
FFT_SIZE = 2048                         # How many points for FFT
WINDOW = hamming(FRAME_LEN)             # Window function
PRE_EMPH = 0.95                         # Pre-emphasis factor

BANDS = 40                              # Number of Mel filters
COEFS = 26                              # Number of Mel cepstra coefficients to keep
POWER_SPECTRUM_FLOOR = 1e-100           # Flooring for the power to avoid log(0)
# M, CF = melfb(BANDS, FFT_SIZE, FS)      # The Mel filterbank matrix and the center frequencies of each band
D = dctmtx(BANDS)[1:COEFS+1]            # The DCT matrix. Change the index to [0:COEFS] if you want to keep the 0-th coefficient
invD = inv(dctmtx(BANDS))[:,1:COEFS+1]  # The inverse DCT matrix. Change the index to [0:COEFS] if you want to keep the 0-th coefficient

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """
    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]
    
    if overlap >= length:
        raise Exception( "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise Exception("overlap must be nonnegative and length must "\
                          "be positive")

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise Exception( "Not enough data points to segment array in cut mode; try pad or wrap")
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
        


def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]
#         print ("i:",low, " c:",cen, " h:", hi)

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)

        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])
#         print (fbank)
    return fbank, freqs

def mfcc(input, nwin=256, nfft=512, fs=16000, nceps=COEFS):
    """Compute Mel Frequency Cepstral Coefficients.

    Parameters
    ----------
    input: ndarray
        input from which the coefficients are computed

    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.

    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum

    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""

    # MFCC parameters: taken from auditory toolbox
    overlap = nwin - 160
    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level)
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
#     nfil = nlinfil + nlogfil
    w = hamming(nwin, sym=0)

#     pylab.plot( w )
#     pylab.show()
    
    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]
    
    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    
    framed = segment_axis(extract, nwin, overlap) * w

    # Compute the spectrum magnitude
    spec = np.abs(fft(framed, nfft, axis=-1))
#     print(spec.shape , "   ",fbank.T.shape)
    # Filter the spectrum through the triangle filterbank
    
    mspec = np.log10(np.dot(spec, fbank.T))

#     sfbank = np.dot(spec, fbank.T)
#     sfbank = sum(sfbank)
#     mspec = np.log10( sfbank)
    

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]
    
#     ceps = dct(mspec, type=2)
    print("ceps :", ceps.shape, "  ", mspec.shape,"  ", spec.shape)
    return ceps, mspec, spec

def preemp(input, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)




def show_MFCC(mfcc):

    imshow(mfcc.T, aspect="auto", interpolation="none")
    title("MFCC features")
    xlabel("Frame")
    ylabel("Dimension")
    show()

def show_MFCC_spectrum(mfcc):
    """
    Show the spectrum reconstructed from MFCC as an image.
    """
    imshow(dot(invD, mfcc.T), aspect="auto", interpolation="none", origin="lower")
    title("MFCC spectrum")
    xlabel("Frame")
    ylabel("Band")
    show()


def getCepsVect(y):
    ceps, mspec, spec = mfcc(y)
    vect_of_mccf = np.zeros(COEFS)
    for j in range(COEFS): 
        vect_of_mccf[j] =  RecordModule.arithmeticMean(ceps.T[j]) 

    vect_of_mccf[0] = 0 
    return vect_of_mccf


if __name__ == '__main__':
    
# ++++++++++++++++++++++++++++++++++++++++++++   
  for i in range(11):
    print("please speak a word into the microphone")
    filename = "learn_set//wlacz//"+str(i+1)+".wav"
#     RecordModule.record_to_file(filename)
#     print("done - result written to ", filename)
#     filename = 'learn_set//wlacz//3.wav'

# ++++++++++++++++++++++++++++++++++++++++++++
    t, extract = PlotModule.readWav(filename, FS)
    extract = RecordModule.preemp(extract)
    
    fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y = RecordModule.detectSingleWord(t,extract)


    ceps, mspec, spec = mfcc(word_y)
    
    
    vect_of_mccf = np.zeros(len(ceps.T))
    
    for i in range(len(ceps.T)): 
        vect_of_mccf[i] =  max(ceps.T[i]) # sum(data.T[i]) #
    


    pylab.title("ceps : ") 
    pylab.plot(range(len(vect_of_mccf)), vect_of_mccf, 'g')
  pylab.show()

#     show_MFCC_spectrum(ceps)
    