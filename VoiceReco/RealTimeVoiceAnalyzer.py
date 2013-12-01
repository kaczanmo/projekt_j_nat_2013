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
from numpy.ma.core import arange

'''
Created on 23-10-2013

@author: Tenac
'''
WYKRES = False
MODE = "TEST"
# MODE =  "ACTIVE"
#/////////////////////////////////////



num_ceps = 13*3

    
def saveMfccMatrixToFile(filename, mfccMatrix):
    myFile = open(filename, 'w')
    
    for i in range(len(mfccMatrix)): 
        myFile.write(str(sum(mfccMatrix[i]))+'\n')
    myFile.close()
    
def fSimilatiry( v1, v2 ):  
    E = 1.0
    ret = 0.0
   
    param = 0
    if(param == 0): # euklidesowa odl     
        for j in range(len(v1)): #Euklides
            ret += (v1[j] - v2[j])**2
        ret = math.sqrt(ret)   
    if (param == 1): # hamming odl     
        for j in range(len(v1)): #Hamming
            ret += abs(np.array(v1[j]) - np.array(v2[j]))  
#     print("  VAL SUM     ", ret)
    ret = 1.0/(ret+E)  
   
    return ret 

def fSimilatiryMatrix( m1, m2 ):  
    ret = 0.0
    for j in range(10):
        ret += fSimilatiry(m1[j], m2[j])
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
#     ceps_vect  = MelFeat.calcMelVectFeatures(ceps_vect)
    

    return ceps_vect

      
    
def readTrainSpeeches(path, numOfSpeech):
    learned_ceps = [[0] for x in range(numOfSpeech)]
    for i in range(numOfSpeech):
        ceps = getCepsVectFromFile(""+path+str(i+1)+'.wav')
    ##########################
        learned_ceps[i] = ceps
       
    return learned_ceps   
            
   

def meanLearnedCeps(lr, numOfSpeech):
     mean_of_mccf = [[0]*MelFeatures.numcepsBands for x in range(MelFeatures.numallceps)]
     
     for k in range(numOfSpeech):
         for i in range(MelFeatures.numallceps):
             for j in range(MelFeatures.numcepsBands):
                 mean_of_mccf[i][j] += lr[k][i][j]
                 
     for i in range(MelFeatures.numallceps):
        for j in range(MelFeatures.numcepsBands):
            mean_of_mccf[i][j] = mean_of_mccf[i][j] / numOfSpeech   
     return mean_of_mccf   

        




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
            dists.append( [fSimilatiryMatrix(predict, learned_speech_tab[i].learned_ceps[g]) , [learned_speech_tab[i].uniqueId]])
            
        
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
             maxForCommand[i] = [fSimilatiryMatrix(predict, learned_speech_tab[i].learned_ceps_abs)]
             

    
#     print("[WLACZ:", max_wlacz,"][WYLACZ:",max_wylacz,"][PODGLOS:",max_podglos,"][SCISZ:",max_scisz,"]" )
    ret = np.zeros(len(learned_speech_tab))
    max = numpy.max(maxForCommand)
    for i in range(len(learned_speech_tab)):
        if maxForCommand[i] == max:
#             print("#"+learned_speech_tab[i].name+" ,", max)
            ret[i] = 1
    return ret         
        
def getClasificationDecision(t, y):
    print("predict? ...")
    predict =  getCepsVectFromData(t, y)

    NNRes = nearestNeighbour(predict)
    NMRes = nearestMean(predict)
    NANRes = nearestAlfaNeighbour(predict, 4)
    ALLRes = np.zeros(len(learned_speech_tab))
    
    for i in range(len(learned_speech_tab)):
        ALLRes[i] = NNRes[i] + NMRes[i]  + NANRes[i] ##
        ALLRes[i] = ALLRes[i]/3.0*100.0
        
    print("RESULT:")
    for i in range(len(learned_speech_tab)):
        print(learned_speech_tab[i].name , "  ", int(ALLRes[i]), " %")  
        
    ai=0
    max = -1
    for i in range(len(learned_speech_tab)):  
        if ALLRes[i] > max :
            ai = i
            max = ALLRes[i]        
        
    return ai    
        
def go():   
    print("please speak a word into the microphone")
    t, y = RecordModule.getSpeechFromMic()
    print("done")
#     t,y = PlotModule.readWav("learn_set//wylacz//9.wav", 44100.0)
    
    getClasificationDecision(t, y)
    


    print("done")        
        
        
def goTest():
 
    stats = []
 
    print("start")
    for i in range(len(learned_speech_tab)):
        okey=0
        bad=0    
        for j in range(learned_speech_tab[i].numOfSpeech):
            print("predict? ...")
            predict = learned_speech_tab[i].learned_ceps[j]
            learned_speech_tab[i].learned_ceps[j] =  [[-9999]*MelFeatures.numcepsBands for x in range(MelFeatures.numallceps)]
#             print (learned_speech_tab[i].learned_ceps[j])
            NNRes = nearestNeighbour(predict)
            NMRes = nearestMean(predict)
            NANRes = nearestAlfaNeighbour(predict, 4)
            ALLRes = np.zeros(len(learned_speech_tab))
            
            for ij in range(len(learned_speech_tab)):
                ALLRes[ij] = NNRes[ij] + NMRes[ij]  + NANRes[ij] ##
                ALLRes[ij] = ALLRes[ij]/3.0*100.0
                
            print("RESULT:")
            for ij in range(len(learned_speech_tab)):
                print(learned_speech_tab[ij].name , "  ", int(ALLRes[ij]), " %")  
                
            ai=0
            max = -1
            for ij in range(len(learned_speech_tab)):  
                if ALLRes[i] > max :
                    ai = ij
                    max = ALLRes[ij]        
            
            learned_speech_tab[i].learned_ceps[j] = predict
            if learned_speech_tab[ai].name == learned_speech_tab[i].name:
                okey+=1
            else:
                bad+=1
        stats.append([learned_speech_tab[i].name,okey,bad,learned_speech_tab[i].numOfSpeech] )
          
              

    print(stats)
    sum_ok = 0.0
    sum_bad = 0.0
    for i in range(len(stats)):
        sum_ok+=stats[i][1]     
        sum_bad+=stats[i][2]

    rate =  (sum_ok/(sum_ok+sum_bad))*100.0
    print("skutecznosc:",(rate),"%")
    print("done") 
       
       
def goRecord():
    global learned_speech_tab 
    learned_speech_tab = []
    learned_speech_tab.append(VoiceCommand("WLACZ", "learn_set//wlacz//", 15, 0))
    learned_speech_tab.append(VoiceCommand("WYLACZ", "learn_set//wylacz//", 15, 1))
    learned_speech_tab.append(VoiceCommand("SCISZ", "learn_set//scisz//", 10, 2))
    learned_speech_tab.append(VoiceCommand("PODGLOS", "learn_set//podglos//", 10, 3))
    
   
    
    for i in range(len(learned_speech_tab)):
        print(learned_speech_tab[i].name)
        (learned_speech_tab[i].learned_ceps) = readTrainSpeeches(learned_speech_tab[i].folderPath, learned_speech_tab[i].numOfSpeech )
        learned_speech_tab[i].learned_ceps_abs = meanLearnedCeps(learned_speech_tab[i].learned_ceps, learned_speech_tab[i].numOfSpeech )
    

    
   
    if(MODE == "ACTIVE"):
            while True:
                 go()
    elif(MODE == "TEST"):
            goTest()    
 


if __name__ == '__main__':
     print ("STARTING!")
     if(MODE == "ACTIVE"):
            threading.Thread(goRecord()).start()
     elif(MODE == "TEST"):
            goRecord()  
    
     
     

     