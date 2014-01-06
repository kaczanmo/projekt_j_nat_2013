'''
Created on 23-10-2013

@author: Tenac
'''

from array import array
from struct import pack
from sys import byteorder
import copy
import pyaudio
import wave
import pylab
import numpy
import math

from mealfeat import MelFeatures
import PlotModule
from scipy.signal import lfilter



RATE = 44100
THRESHOLD = 300  # audio levels not normalised.
CHUNK_SIZE = 1024
SILENT_CHUNKS = int(0.15 * RATE / CHUNK_SIZE)  # about x sec
SPEECH_CHUNKS = int(0.1 * RATE / CHUNK_SIZE)  # about x sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
MAX_LENGTH_SEC = int(2.0 * RATE / CHUNK_SIZE)  # about x sec

CHANNELS = 1
TRIM_APPEND = RATE / 4

def is_silent(data_chunk):
    """zwraca TRUE jest probka ramka dzwieku przekroczy prog ciszy"""
    return max(data_chunk) < THRESHOLD

def normalize(data_all):
    """Normalizacja glosnosci dzwieku do max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r


def trim(data_all):
    ''' przyciecie sygnalu dzwiekowego - wyciecie ciszy  z przodu i z tylu pliku '''
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[int(_from):int(_to + 1)])

def record():
    """funkcja nagrywa dzwiek z mikrofinu i zwraca
    jako tablice"""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    silent_chunks = 0
    speech_chunks = 0
    audio_started = False
    data_all = array('h')
#     print ("SIL CH : ", SILENT_CHUNKS, "  SPE CH : ", SPEECH_CHUNKS)
    while True:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)

        silent = is_silent(data_chunk)
       
        if audio_started:
#             print ("si c:", silent_chunks, "  sp c:", speech_chunks)
            if not silent:
                silent_chunks = 0
                speech_chunks += 1
                if speech_chunks > MAX_LENGTH_SEC :
                    break
            
            if silent :
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS : #
                    if speech_chunks > SPEECH_CHUNKS:
                        break
                    else:
                        data_all = data_all[len(data_all) - silent_chunks : len(data_all)]  #array('h')
                        speech_chunks = 0
                        silent_chunks = 0
                        audio_started = False
                        
           
        elif not silent:
            audio_started = True
            speech_chunks = 0
            silent_chunks = 0

            

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    data_all = normalize(data_all)
    return sample_width, data_all

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    
    data = pack('<' + ('h' * len(data)), *data)
    
    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()
    
def getSpeechFromMic():
    "nagrywanie dzwieku z mikrofonu "
    sample_width, data = record()
    
#     data = pack('<' + ('h' * len(data)), *data)

    y = numpy.atleast_2d(data)[0]
    timp=len(y)/RATE
    t=numpy.linspace(0,timp,len(y))
    return t, y
   
    
def arithmeticMean(iterable):
    ''' srednia arytmetyczna '''
    return sum(iterable)/float(len(iterable))
 
def sampleStandardDeviation(iterable):
    '''
    odchylenie std
    '''
    l=len(iterable)
    x2=[x**2 for x in iterable]
    return math.sqrt(float(l)/(l-1)*(arithmeticMean(x2)-arithmeticMean(iterable)**2))

    
def detectSingleWord(t,y):
    '''
    alg detekcji pojedynczego slowa (komendy) slowa
    '''
    framelen = 10 # frame length in ms
    framesamples = int(RATE/1000*framelen)
    frames = int(len(y) / framesamples)
    wordspower =  numpy.zeros(frames) # array('d' , range(frames))  #
    
    
    for i in range(frames):
        wordspower[i] = (sum(abs( y[i*framesamples+1:(i+1)*framesamples])     ) )
       
    fr = range(frames)

    ysigned = numpy.zeros(len(y))
    for i in range(len(ysigned)):
       if(y[i] >= 0): 
           ysigned[i] = 1
       else:
           ysigned[i] = -1
    
    wordszeros = numpy.zeros(int(len(ysigned) / framesamples))
    for i in range(frames):
        wordszeros[i] = 0.5 * sum(abs(numpy.subtract(ysigned[(i*framesamples)+1:(i+1)*framesamples] , ysigned[(i*framesamples):(i+1)*framesamples-1])))   #sum(abs(y[i*frames:(i+1)*frames]))
#         print(i , " p p 0 : ",wordszeros[i])
        
    IMN = arithmeticMean(wordspower[0:int(100/framelen)]) #int(100/framelen)
    IMX = max(wordspower[0:int(100/framelen)])  #
    I1 = 0.03*(IMX - IMN) + IMN
    I2 = 4 * IMN 
    ITL = min(I1, I2)
    ITU = 5*ITL
      
    IZC = numpy.mean(wordszeros)
    IZCsigma = sampleStandardDeviation(wordszeros)

    IZCT = min(25.0/10.0*framelen, IZC + 2*IZCsigma)

    wordstart = 0
    wordstop = 0

    wordsdetect = [] # numpy.zeros(int(len(y)))

    for i in range(len(y)):
        wordsdetect.append(0.01) 
    
    
#     print("ITU", ITU , "  ITL:", ITL)
#     print("I1", I1 , "  I2:", I2)
#     print("IZCT", IZCT )
    
    
    # -- >>> od 0 do wordstart
    for m in range(0,frames,1):
        if(wordstart > 0): 
            break
#         print ("? : ",m, "  ",  wordspower[m] , "  ITL:", ITL )
        if(wordspower[m] >= ITL ):
            for i in range(m, frames, 1):
                if(wordspower[i] < ITL ):
                    break
                else:
                    if(wordspower[m] >= ITU ): #and arithmeticMean(wordspower[i+1:i+1+(int(200/framelen))]) >= ITU
#                         print("OK",i, ",",m,",",  wordspower[i] , "  ITU:", ITU  )
                        wordstart = i
                        if(i == m):
                            wordstart = wordstart-1
                        break    
  
                       
                            
#     #IZCT    <<< --- cofniecie przed wordstart
    hi = wordstart
    for i in range(wordstart, 2, -1):
#         print ("?IZCT : ",i, "  ",  wordszeros[i] , "  " , IZCT, "  ", (framelen*(i - wordstart)))
        if(wordszeros[i] >= IZCT ):   
            hi  = i
        if(hi< wordstart and wordszeros[hi-1]>=IZCT ) :
            wordstart = hi-1
            break    

#     IMN = arithmeticMean(wordspower[frames-(int(100/framelen)):frames])
#     IMX = max(wordspower[frames-(int(100/framelen)):frames]) 
#     I1 = 0.03*(IMX - IMN) + IMN
#     I2 = 4 * IMN 
#     ITL = min(I1, I2)
#     ITU = 5*ITL

    # <<< --- od konca do wordstop
    for m in range(frames-2,0,-1):
            if(wordstop > 0): 
                break
#             print ("? : ",m, "  ",  wordspower[m] , "  ITL:", ITL )
            if(wordspower[m] >= ITL ):
                for i in range(m, 0, -1):
                    if(wordspower[i] < ITL):
                        break
                    else:
                        if(wordspower[m] >= ITU):
#                             print("OK",i, ",",m,",",  wordspower[i] , "  ITU:", ITU  )
                            wordstop = i
                            if(i == m):
                                wordstop = wordstop+1
                            break  



    #IZCT    --->>> dalej za wordstop
    hi = wordstop
    for i in range(wordstop, frames-2, 1):
#         print ("?IZCT : ",i, "  ",  wordszeros[i] , "  " , IZCT, "  ", (framelen*(i - wordstop)))
        if(wordszeros[i] >= IZCT ):   
            hi  = i
        if(hi> wordstop and wordszeros[hi+1]>=IZCT ) :       #and (150>(i - wordstop)*framelen)
            wordstop = hi+1
            break        
        
    if(wordstart >= wordstop or (wordstop-wordstart)*framelen < 50 ):
        wordstart = 0
        wordstop = frames
#     print("L:", wordstart, "  P:", wordstop) 
     
        
    for i in range(wordstart*(framesamples) , wordstop*(framesamples), 1):
            wordsdetect[i] = 1.0 

    word_fr = t[wordstart*(framesamples) : wordstop*(framesamples)]
    word_y = y*wordsdetect
    word_y = word_y[wordstart*(framesamples) : wordstop*(framesamples)]
        
#     print('ITL:',ITL, 'ITU:',ITU)      
    return fr, wordspower, wordszeros, wordsdetect, ITL, ITU,  word_fr, word_y
    
def preemp(input, p=0.95):
    """Pre-emphasis filter."""
#     return lfilter([1., -p], 1, input) 
#     return numpy.append(input[0],input[1:]-p*input[:-1])
    y2 = input
    y2[0]=0
    for i in range(1,len(input)-1,1):
        y2[i]=input[i]-(p*y2[i-1])
           
#     return lfilter(numpy.array([1,-p]),1,input)
    return y2
    
if __name__ == '__main__':
    print("please speak a word into the microphone")
#     filename = "learn_set//podglos//"+str( 4 )+".wav"   
    filename = 'test.wav' 
#     record_to_file(filename)
#     print("done - result written to ", filename)
#     t,y = PlotModule.readWav(filename, RATE)
    
    t,y = getSpeechFromMic()
    print("done")
    
#     t,y = getSpeechFromMic()
    y = preemp(y)
    fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y = detectSingleWord(t,y)
     
 
    pylab.subplot(611)
    pylab.title(filename) 
    pylab.plot(t, y, 'b')
     
    pylab.subplot(612)
    pylab.plot(fr, (wordspower), 'r')
 
    arrITL = array('f', range(len(wordspower)) )
    for i in range(len(wordspower)):
        arrITL[i] = ITL
    pylab.plot(fr, arrITL, 'g')
     
    arrITU = array('f', range(len(wordspower)) )
    for i in range(len(wordspower)):
        arrITU[i] = ITU
    pylab.plot(fr, arrITU, 'r')
     
    pylab.subplot(613)
    pylab.plot(fr, wordszeros, 'r')
     
    pylab.subplot(614)
    pylab.plot(t, (wordsdetect ), 'g')
     
    pylab.subplot(615)
    pylab.plot(word_fr, (word_y ), 'r')
    
    
    
#     ceps = MfccModule.getCepsVect(word_y)
#     MelFeat = MelFeatures()
#     rawdata = MelFeat.loadWAVfile(filename)
#     MFCC    = MelFeat.calcMelMatrixFeatures(word_y)          
      
#     MelFeat.plotSpectrogram(MFCC)
        
    
    
#     pylab.subplot(616)
#     pylab.title("MFCC : ") 
#     pylab.plot(range(len(MFCC)) , MFCC,'g')

     
     
    pylab.show()
 
     
     
    
    
    