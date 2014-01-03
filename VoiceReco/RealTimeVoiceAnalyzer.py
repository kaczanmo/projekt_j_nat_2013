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

import pylab
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
RECALCULATE_CEPS_MATRIX = False # recalculate and save to file
prog_Komenda_nieznana = 71.0 # %
#/////////////////////////////////////


    
def writeMfccMatrixToTxtFile(filename, mfccMatrix):
    myFile = open(filename, 'w')
    for kk in range(len(mfccMatrix)):
        for ll in range(len(mfccMatrix[0])) :
              myFile.write(str(mfccMatrix[kk][ll])+'\n')
    myFile.close()
    
def readMfccMatrixToTxtFile(filename):
#     mfccMatrix = [[0 for x in range(5)] for y in range(39)]
#     myFile = open(filename, 'r')
#     for kk in range(39):
#         for ll in range(5) :
#              line = myFile.read().split()
#              mfccMatrix[kk][ll] = line # (str(mfccMatrix[kk][ll])+'\n')
#              print (mfccMatrix[kk][ll])
#     print (mfccMatrix)         
#     myFile.close()  
    kk=-1
    nr=0
    mfccMatrix = [[0]*MelFeatures.numcepsBands for x in range(MelFeatures.numallceps)]
    with open(filename, 'r') as f:
        for line in f:
            numbers_float =  line.split()
#             if nr%MelFeatures.numallceps
            if nr%MelFeatures.numcepsBands == 0 :
                kk+=1
#             print (numbers_float[0])
            mfccMatrix[kk][nr%MelFeatures.numcepsBands] = (float(numbers_float[0]))
            nr+=1
            
    return mfccMatrix
            
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
    for j in range(len(m1)):
        ret += fSimilatiry(m1[j], m2[j])
    return ret 

def getCepsMatrixFromWavFile(filename):
    Fs = 44100.0;  # sampling rate
    t,y = PlotModule.readWav(filename, Fs)
    ##########
    ceps_matrix = getCepsMatrixFromData(t,y)
    return ceps_matrix 


def getCepsMatrixFromData(t,y):
    ##########
    y = RecordModule.preemp(y)
    fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y = RecordModule.detectSingleWord(t,y)
    ##################
    MelFeat = mealfeat.MelFeatures()
    ceps_matrix    = MelFeat.calcMelFeatures(word_y)

    return ceps_matrix

      
    
def readTrainSpeeches(path, numOfSpeech):
    learned_ceps = [[0] for x in range(numOfSpeech)]
    ceps = []
    for i in range(numOfSpeech):
        if RECALCULATE_CEPS_MATRIX == True :
            ceps = getCepsMatrixFromWavFile(""+path+str(i+1)+'.wav')
            writeMfccMatrixToTxtFile(""+path+str(i+1).rsplit( ".", 1 )[ 0 ]+'.txt', ceps)
        else:
            ceps = readMfccMatrixToTxtFile(""+path+str(i+1).rsplit( ".", 1 )[ 0 ]+'.txt' )
#         print(ceps)    
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
        
def getClasificationDecision(predict):
    NNRes = nearestNeighbour(predict)
    NMRes = nearestMean(predict)
    NANRes = nearestAlfaNeighbour(predict, 20)
    ALLRes = np.zeros(len(learned_speech_tab))
    
    for i in range(len(learned_speech_tab)):
        ALLRes[i] = (0.2*NNRes[i])+ (0.5*NMRes[i])  + (0.3*NANRes[i]) ## 0.1  0.5  0.4 -> 84% 
        ALLRes[i] = ALLRes[i]*100.0
        
    print("RESULT:")
    for i in range(len(learned_speech_tab)):
        print(learned_speech_tab[i].name , "  ", int(ALLRes[i]), " %")  
        
    ai=0
    max_val = -1
    for i in range(len(learned_speech_tab)):  
        if ALLRes[i] > max_val :
            ai = i
            max_val = ALLRes[i]        
    
    # wskazuje na poza tablice, czyli przypodzadkowanie komendy jako nieznana
    if(max_val < prog_Komenda_nieznana):
        ai = len(learned_speech_tab)+1
    return ai    
        
def go():   
    print("please speak a word into the microphone")
    t, y = RecordModule.getSpeechFromMic()
    print("done")
#     t,y = PlotModule.readWav("learn_set//wylacz//9.wav", 44100.0)
    predict =  getCepsMatrixFromData(t, y)
    getClasificationDecision(predict)
    


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
            
            learned_speech_tab[i].learned_ceps[j] =   [[99999 for x in range(MelFeatures.numcepsBands)] for y in range(MelFeatures.numallceps)]

            ai = getClasificationDecision(predict)      
            
            learned_speech_tab[i].learned_ceps[j] = predict  
            if ai != (len(learned_speech_tab)+1) and learned_speech_tab[ai].name == learned_speech_tab[i].name:
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
    print("skutecznosc: %3.2f" % (rate) , "%")
    print("done") 
       
       
def goRecord():
    global learned_speech_tab 
    learned_speech_tab = []
    learned_speech_tab.append(VoiceCommand("WLACZ", "learn_set//wlacz//", 15, 0))
    learned_speech_tab.append(VoiceCommand("URUCHOM", "learn_set//uruchom//", 40, 1))
    learned_speech_tab.append(VoiceCommand("WYLACZ", "learn_set//wylacz//", 15, 2))
    learned_speech_tab.append(VoiceCommand("SCISZ", "learn_set//scisz//", 25, 3)) 
    learned_speech_tab.append(VoiceCommand("PODGLOS", "learn_set//podglos//", 25, 4)) 
    learned_speech_tab.append(VoiceCommand("NASTEPNY", "learn_set//nastepny//", 25, 5))
    learned_speech_tab.append(VoiceCommand("POPRZEDNI", "learn_set//poprzedni//", 20, 6))
      
    learned_speech_tab.append(VoiceCommand("WYCISZ", "learn_set//wycisz//", 20, 7))
    learned_speech_tab.append(VoiceCommand("JEDYNKA", "learn_set//jedynka//", 20, 8))
    learned_speech_tab.append(VoiceCommand("DWOJKA", "learn_set//dwojka//", 25, 9))
    
    # wazne
    global noneVoiceCommand
    noneVoiceCommand = VoiceCommand("NIEZNANA", "", 0, 999)
    
   
    
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
    
     
     

     