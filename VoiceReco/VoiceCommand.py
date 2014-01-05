'''
Created on 24-11-2013

@author: Tenac
'''


class VoiceCommand(object):
    '''
    VoiceCommand przechowuje informacje o komendzie, jej nagranych probkach i wspolcznnikach MFCC
    '''
    name = ''
    folderPath = '' 
    numOfSpeech = 0
    uniqueId = 0
    learned_ceps = []
    learned_ceps_abs = []

    def __init__(self, qname, qfoldPath, qnum, quninq):
        '''
        Constructor
        '''
        self.name = qname
        self.folderPath = qfoldPath
        self.numOfSpeech = qnum
        self.uniqueId = quninq
        
        
    def getName(self):
        return self.name
