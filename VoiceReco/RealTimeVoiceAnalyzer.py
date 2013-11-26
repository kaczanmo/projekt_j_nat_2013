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
from mealfeat import MelFeatures
from VoiceCommand import VoiceCommand

'''
Created on 23-10-2013

@author: Tenac
'''
num_ceps = 39
# num_train_speech = 11



ANN = False
WYKRES = False

    
    
def saveMfccMatrixToFile(filename, mfccMatrix):
    myFile = open(filename, 'w')
    
    for i in range(len(mfccMatrix)): 
        myFile.write(str(sum(mfccMatrix[i]))+'\n')
    myFile.close()
    
def fSimilatiry( v1, v2 ):  
    E = 1.0
    ret = 0.0
    
#     for j in range(num_ceps): #Euklides
#         ret += (v1[j] - v2[j])**2
#     ret = math.sqrt(ret)   
    
    for j in range(len(v1)): #Hamming
        ret += abs(v1[j] - v2[j])  
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
    #################
    #ceps_vect = MelFeatures.c  MelFeat.calcMelVectFeatures(MFCC) #MfccModule.getCepsVect(word_y)
    
    ##################
    MelFeat = mealfeat.MelFeatures()
    ceps_vect    = MelFeat.calcMelFeatures(word_y)
    ceps_vect  = MelFeat.calcMelVectFeatures(ceps_vect)
    
    print("FDFD", ceps_vect.shape)
    return ceps_vect

      
    
def readTrainSpeeches(path, numOfSpeech):
    learned_ceps = [[0]*numOfSpeech for x in range(num_ceps)]
    for i in range(numOfSpeech):
        ceps = getCepsVectFromFile(""+path+str(i+1)+'.wav')
    ##########################
        vect_of_mccf = np.zeros(num_ceps)
        for j in range(num_ceps): 
            vect_of_mccf[j] =  ceps[j]
        learned_ceps[i]= vect_of_mccf
        
    return learned_ceps   
            
   

def meanLearnedCeps(lr, numOfSpeech):
     vect_of_mccf = np.zeros(num_ceps)
     for i in range(numOfSpeech):
        ceps = lr[i]
        for j in range(num_ceps): 
            vect_of_mccf[j] +=  ceps[j]

     for j in range(num_ceps): 
         vect_of_mccf[j] = vect_of_mccf[j] / numOfSpeech   
     return vect_of_mccf   

        




def nearestNeighbour(predict):
    print("nearestNeighbour")
    res = nearestAlfaNeighbour(predict, 1)
    return res
        
def nearestAlfaNeighbour(predict, alfa):
    print("nearestAlfaNeighbour")
    
#     if alfa > num_train_speech:
#         alfa = num_train_speech
    
    allTrainSpeech = 0

    for i in range(len(learned_speech_tab)):
        allTrainSpeech += learned_speech_tab[i].numOfSpeech
        
    
    dists = []

    for i in range(len(learned_speech_tab)):
        for g in range(0,learned_speech_tab[i].numOfSpeech):
            dists.append( [fSimilatiry(predict, learned_speech_tab[i].learned_ceps[g]) , [learned_speech_tab[i].uniqueId]])
            
        
        
    dists = sorted(dists, reverse=True) 
    print(dists)

    maxForCommand = np.zeros(len(learned_speech_tab))
    
    for i in range(len(learned_speech_tab)):
        alfaPom = 0
        if alfa > learned_speech_tab[i].numOfSpeech :
            alfaPom = learned_speech_tab[i].numOfSpeech
        else :
            alfaPom = alfa
        max = 0      
        for g in range(0,int(alfaPom)):  
            if(dists[g][1] == [learned_speech_tab[i].uniqueId]): 
                max +=1
        maxForCommand[i] = max/alfaPom
    
    ret = np.zeros(len(learned_speech_tab))
    for i in range(len(learned_speech_tab)):  
        ret[i] = maxForCommand[i]
  
    return ret
        

def nearestMean(predict):
    print("nearestMean")

    maxForCommand = [ []for x in range(len(learned_speech_tab))]

    for i in range(len(learned_speech_tab)):
             maxForCommand[i] = [fSimilatiry(predict, learned_speech_tab[i].learned_ceps_abs)]
             

    
#     print("[WLACZ:", max_wlacz,"][WYLACZ:",max_wylacz,"][PODGLOS:",max_podglos,"][SCISZ:",max_scisz,"]" )
    ret = np.zeros(len(learned_speech_tab))
    max = numpy.max(maxForCommand)
    for i in range(len(learned_speech_tab)):
        if maxForCommand[i] == max:
#             print("#"+learned_speech_tab[i].name+" ,", max)
            ret[i] = 1
    return ret         
        
def go():   

    
    print("please speak a word into the microphone")
    t, y = RecordModule.getSpeech()
    print("done")
#     t,y = PlotModule.readWav("learn_set//wlacz//10.wav", 44100.0)
    
    
    print("predict? ...")
    predict =  getCepsVectFromData(t, y)

    NNRes = nearestNeighbour(predict)
    NMRes = nearestMean(predict)
    NANRes = nearestAlfaNeighbour(predict, 4)
    ALLRes = np.zeros(len(learned_speech_tab))
    
    for i in range(len(learned_speech_tab)):
        ALLRes[i] = NNRes[i] + NMRes[i] + NANRes[i]
        ALLRes[i] = ALLRes[i]/3.0*100.0
        
    print("RESULT:")
    for i in range(len(learned_speech_tab)):
        print(learned_speech_tab[i].name , "  ", int(ALLRes[i]), " %")
    
    if ANN:
        bpnn.test( [[predict, [1,1,1,1]]])
        
    if WYKRES:  
        pylab.subplot(111)
        for i in range(len(learned_speech_tab)):  
            pylab.plot(range(num_ceps), learned_speech_tab[i].learned_ceps_abs, 'r' )    

            
        pylab.subplot(111)
        pylab.title("porownanie") 
        pylab.plot(range(num_ceps), predict[:num_ceps], 'k' ) 
        
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
    global learned_speech_tab 
    learned_speech_tab = []
    learned_speech_tab.append(VoiceCommand("WLACZ", "learn_set//wlacz//", 11, 0))
    learned_speech_tab.append(VoiceCommand("WYLACZ", "learn_set//wylacz//", 11, 1))
    learned_speech_tab.append(VoiceCommand("SCISZ", "learn_set//scisz//", 11, 2))
    learned_speech_tab.append(VoiceCommand("PODGLOS", "learn_set//podglos//", 11, 3))
    
    for i in range(len(learned_speech_tab)):
        print(learned_speech_tab[i].name)
        learned_speech_tab[i].learned_ceps = readTrainSpeeches(learned_speech_tab[i].folderPath, learned_speech_tab[i].numOfSpeech )
        learned_speech_tab[i].learned_ceps_abs = meanLearnedCeps(learned_speech_tab[i].learned_ceps, learned_speech_tab[i].numOfSpeech )
    

    
################################
    global bpnn
    if ANN: 
        allTrainSpeech = 0
        for i in range(len(learned_speech_tab)):
            allTrainSpeech += learned_speech_tab[i].numOfSpeech
        
        bpnn = Bpnn.NN(num_ceps, 4, 4)
        pat = [ [[0]*num_ceps,[0]*4 ]for x in range(allTrainSpeech)]
        
#         for i in range(len(learned_speech_tab)):
#              pat[i] = [learned_speech_tab[i].learned_ceps, [1,0,0,0]]
     
#         for i in range(num_train_speech):
#             pat[i] = [learned_ceps_wlacz[i], [1,0,0,0]]
#         for i in range(num_train_speech):
#             pat[i+1*num_train_speech] = [learned_ceps_wylacz[i], [0,1,0,0]]
#         for i in range(num_train_speech):
#             pat[i+2*num_train_speech] = [learned_ceps_podglos[i], [0,0,1,0]]
#         for i in range(num_train_speech):
#             pat[i+3*num_train_speech] = [learned_ceps_scisz[i], [0,0,0,1]]
             
        print(pat)    
             
        bpnn.train(pat)
        # test it
        bpnn.test(pat)
    
    while True:
        go()
 


if __name__ == '__main__':
     print ("STARTING!")
     threading.Thread(goRecord()).start()
     
     

     