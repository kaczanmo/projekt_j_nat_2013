'''
Created on 25-10-2013

@author: Tenac
'''
import PlotModule
import RecordModule
import threading
import numpy as np

import mealfeat

import numpy
import math

import MfccModule
import pylab
import MfccModule2
import Bpnn

'''
Created on 23-10-2013

@author: Tenac
'''
num_ceps = 39
num_train_speech = 10

train_speech_tab = [["", 2, "WLACZ"] 
                    ]


ANN = False

    
    
def saveMfccMatrixToFile(filename, mfccMatrix):
    myFile = open(filename, 'w')
    
    for i in range(len(mfccMatrix)): 
        myFile.write(str(sum(mfccMatrix[i]))+'\n')
    myFile.close()
    
def fSimilatiry( v1, v2 ):  
    E = 1.0
    ret = 0.0
    
    for j in range(num_ceps): #Euklides
        ret += (v1[j] - v2[j])**2
    ret = math.sqrt(ret)   
    
#     for j in range(len(v1)): #Hamming
#         ret += abs(v1[j] - v2[j])  
#     print("  VAL SUM     ", ret)
    ret = 1.0/(ret+E)  
   
    return ret 

def getCepsVectFromFile(filename):
#     Fs = 44100.0;  # sampling rate
#     t,y = PlotModule.readWav(filename, Fs)
# ##########
#     y = RecordModule.preemp(y)
#     fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y = RecordModule.detectSingleWord(t,y)
#    #########################
#     ceps, mspec, spec = MfccModule2.mfcc(word_y)




################
#     MelFeat = mealfeat.MelFeatures()
#     rawdata = MelFeat.loadWAVfile(filename)
#     MFCC    = MelFeat.calcMelFeatures(rawdata)
#     ceps = MFCC
#     ##########################

    Fs = 44100.0;  # sampling rate
    t,y = PlotModule.readWav(filename, Fs)
    ##########
    return getCepsVectFromData(t,y)


def getCepsVectFromData(t,y):
    ##########
    y = RecordModule.preemp(y)
    fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y = RecordModule.detectSingleWord(t,y)
    ceps_vect = MfccModule.getCepsVect(word_y)
    
    print("FDFD", ceps_vect.shape)
    return ceps_vect

      
    
def readTrainSpeeches(path):
    learned_ceps = [[0]*num_train_speech for x in range(num_ceps)]
    for i in range(num_train_speech):
        ceps = getCepsVectFromFile(""+path+str(i+1)+'.wav')
    ##########################
        vect_of_mccf = np.zeros(num_ceps)
        for j in range(num_ceps): 
            vect_of_mccf[j] =  ceps[j]
        learned_ceps[i]= vect_of_mccf
        
    return learned_ceps   
            
   

def meanLearnedCeps(lr):
     vect_of_mccf = np.zeros(num_ceps)
     for i in range(num_train_speech):
        ceps = lr[i]
        for j in range(num_ceps): 
            vect_of_mccf[j] +=  ceps[j]

     for j in range(num_ceps): 
         vect_of_mccf[j] = vect_of_mccf[j] / num_train_speech   
     return vect_of_mccf   

        




def nearestNeighbour(predict):
    print("nearestNeighbour")
    nearestAlfaNeighbour(predict, 1)
        
def nearestAlfaNeighbour(predict, alfa):
    print("nearestAlfaNeighbour")
    
    if alfa > num_train_speech:
        alfa = num_train_speech
    
    dists = [ [2]for x in range(num_train_speech*4)]

    
    for g in range(0,num_train_speech): 
        dists[g] = [fSimilatiry(predict, learned_ceps_wlacz[g]) , [0]]
        
    for g in range(0,num_train_speech): 
        dists[g+1*num_train_speech] = [fSimilatiry(predict, learned_ceps_wylacz[g]) , [1]]
    
    for g in range(0,num_train_speech): 
        dists[g+2*num_train_speech] = [fSimilatiry(predict, learned_ceps_podglos[g])  , [2]]       
    
    for g in range(0,num_train_speech): 
        dists[g+3*num_train_speech] = [fSimilatiry(predict, learned_ceps_scisz[g])  , [3]]    
        
        
        
    dists = sorted(dists, reverse=True) 
    print(dists)

    
    max = 0        
    for g in range(0,alfa): 
        if(dists[g][1] == [0]):
            max +=1
    max_wlacz = max
######################
    max = 0        
    for g in range(0,alfa): 
        if(dists[g][1] == [1]):
           max +=1
    max_wylacz = max
######################
    max = 0        
    for g in range(0,alfa): 
        if(dists[g][1] == [2]):
           max +=1
    max_podglos = max 
######################
    max = 0        
    for g in range(0,alfa): 
        if(dists[g][1] == [3]):
           max +=1
    max_scisz= max    
     
     
    max = numpy.max([max_wlacz, max_wylacz, max_podglos, max_scisz])
     
    print("[WLACZ:", max_wlacz,"][WYLACZ:",max_wylacz,"][PODGLOS:",max_podglos,"][SCISZ:",max_scisz,"]" )
     
    if max_wlacz == max :
        print("#WLACZ ,", max)
    elif max_wylacz == max :
        print("#WYLACZ ,", max)   
    elif max_podglos == max :
        print("#PODGLOS ,", max)     
    elif max_scisz == max :
        print("#SCISZ ,", max)
        
        

def nearestMean(predict):
    print("nearestMean")
    max_wlacz = fSimilatiry(predict, learned_ceps_abs_wlacz)
    max_wylacz = fSimilatiry(predict, learned_ceps_abs_wylacz)
    max_podglos = fSimilatiry(predict, learned_ceps_abs_podglos)
    max_scisz = fSimilatiry(predict, learned_ceps_abs_scisz)


    max = numpy.max([max_wlacz, max_wylacz, max_podglos, max_scisz])
    
    print("[WLACZ:", max_wlacz,"][WYLACZ:",max_wylacz,"][PODGLOS:",max_podglos,"][SCISZ:",max_scisz,"]" )
    
    if max_wlacz == max :
        print("#WLACZ ,", max)
    elif max_wylacz == max :
        print("#WYLACZ ,", max)   
    elif max_podglos == max :
        print("#PODGLOS ,", max)     
    elif max_scisz == max :
        print("#SCISZ ,", max)        
        
def go():   
    WYKRES = False
    
    print("please speak a word into the microphone")
    t, y = RecordModule.getSpeech()
    print("done")
#     t,y = PlotModule.readWav("learn_set//wylacz//10.wav", 44100.0)
    
    
    print("predict? ...")
    predict =  getCepsVectFromData(t, y)

    nearestNeighbour(predict)
    nearestMean(predict)
    nearestAlfaNeighbour(predict, 4)
    if ANN:
        bpnn.test( [[predict, [1,1,1,1]]])
        
    if WYKRES:  
        pylab.subplot(111)   
        pylab.plot(range(num_ceps), learned_ceps_abs_wlacz, 'y' )    
        pylab.plot(range(num_ceps), learned_ceps_abs_wylacz, 'g' )  
        pylab.plot(range(num_ceps), learned_ceps_abs_podglos, 'b' )  
        pylab.plot(range(num_ceps), learned_ceps_abs_scisz, 'b' )  
            
        pylab.subplot(111)
        pylab.title("porownanie") 
        pylab.plot(range(num_ceps), predict[:num_ceps], 'r' ) 
        
        pylab.show()   
            

#   vect_of_mccf = np.zeros(len(learned_ceps))
#     for i in range(train_speech_nr):
#         for j in range(ceps_nr):
#             vect_of_mccf[j] += learned_ceps[i][j]
#     
#     for j in range(ceps_nr): 
#         vect_of_mccf[j] = vect_of_mccf[j]/ceps_nr  

#     pylab.subplot(421)
#     pylab.title(filename) 
#     pylab.plot(t, y)
#      
#     pylab.subplot(422)
#     PlotModule.plotSpectrum(wordsdetect*y,Fs)
#      
#     pylab.subplot(425)
#     pylab.imshow(ceps.T, aspect="auto", interpolation="none")
#     pylab.title("MFCC features")
#     pylab.xlabel("Frame")
#     pylab.ylabel("Dimension")
#      
#     pylab.subplot(426)
#     pylab.imshow(MfccModule2.dot(MfccModule2.invD, ceps.T), aspect="auto", interpolation="none", origin="lower")
#     pylab.title("MFCC spectrum")
#     pylab.xlabel("Frame")
#     pylab.ylabel("Band")
#      
#     pylab.subplot(427)
#      
#    
#         
#     pylab.plot(range(len(vect_of_mccf)) , vect_of_mccf)
#     pylab.xlabel("frames")
#     pylab.ylabel("avg")
#      
#     pylab.show()

    print("done")        
        
        
def go2():
     filename = "b.wav"
     MelFeat = mealfeat.MelFeatures()
     rawdata = MelFeat.loadWAVfile('wlacz.wav')
     MFCC    = MelFeat.calcMelFeatures(rawdata)
     MFCC_s  = MelFeat.calcMelSumsFeatures(MFCC)
    
     MelFeat.plotSpectrogram(MFCC)
     print("done") 
       
       
def goRecord():
    global learned_ceps_wlacz
    global learned_ceps_wylacz
    global learned_ceps_podglos
    global learned_ceps_scisz
    
    global learned_ceps_abs_wlacz
    global learned_ceps_abs_wylacz
    global learned_ceps_abs_podglos
    global learned_ceps_abs_scisz 
    
    learned_ceps_wlacz =  readTrainSpeeches("learn_set//wlacz//")
    learned_ceps_wylacz =  readTrainSpeeches("learn_set//wylacz//")
    learned_ceps_podglos = readTrainSpeeches("learn_set//podglos//")
    learned_ceps_scisz = readTrainSpeeches("learn_set//scisz//") 
    
    learned_ceps_abs_wlacz =  meanLearnedCeps(learned_ceps_wlacz)
    learned_ceps_abs_wylacz =  meanLearnedCeps(learned_ceps_wylacz)
    learned_ceps_abs_podglos = meanLearnedCeps(learned_ceps_podglos)
    learned_ceps_abs_scisz = meanLearnedCeps(learned_ceps_scisz)
    
    global bpnn
    if ANN: 
        bpnn = Bpnn.NN(num_ceps, 4, 4)
        pat = [ [[0]*num_ceps,[0]*4 ]for x in range(num_train_speech*4)]
    
        for i in range(num_train_speech):
            pat[i] = [learned_ceps_wlacz[i], [1,0,0,0]]
        for i in range(num_train_speech):
            pat[i+1*num_train_speech] = [learned_ceps_wylacz[i], [0,1,0,0]]
        for i in range(num_train_speech):
            pat[i+2*num_train_speech] = [learned_ceps_podglos[i], [0,0,1,0]]
        for i in range(num_train_speech):
            pat[i+3*num_train_speech] = [learned_ceps_scisz[i], [0,0,0,1]]
            
        print(pat)    
            
        bpnn.train(pat)
        # test it
        bpnn.test(pat)
    
    while True:
        go()
 


if __name__ == '__main__':
     print ("STARTING!")
     threading.Thread(goRecord()).start()