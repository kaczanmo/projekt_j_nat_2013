import scipy.io.wavfile
import scipy.io as sio
import numpy as np
import math, cmath
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import lfilter
import pylab
import RecordModule
import PlotModule
from array import array

class MelFeatures:
  """Mel-frequency cepstral coefficients

  Written by Mark Harvilla, Michael Garbus, and David Wozny
  """

  ###parameters for feature computation
  a         = 0.97
  t1        = 0.025
  t2        = 0.01
  numFilts  = 15
  minfrq    = 133.0
  maxfrq    = 6000.0
  width     = 1.0
  numcepBasic    = 13
  numallceps = numcepBasic*3
  del_w     = 2.0 #these should be EVEN
  dbl_del_w = 4.0
  numcepsBands = 10

  def __init__(self):
    pass

  def preemph(self,x,a):
      y = lfilter(np.array([1,-a]),1,x)
      return y
  
  def hz2mel(self,frq):
      m   = 2595*math.log(1+frq/700,10)
      return m
  
  def mel2hz(self,m):
      frq = 700*(pow(10,m/2595)-1)
      return frq
  
  def hamming(self,L):
      M  = L-1
      n  = np.arange(0,L)
      hw = 0.54 - 0.46*np.cos(math.pi*2*n/M)
      return hw
  
  def stft(self,x,t1,t2,fs):
      L  = len(x)
  
      N1 = math.floor(t1*fs); nfft = math.pow(2,math.ceil(math.log(N1,2)))
      N2 = math.floor(t2*fs)
  
      numWindows = int(1 + math.floor((L-N1)/N2))
      #correct if all non-full-length windows are dropped
  
      n = range(0,L-1,int(N2))
      W = self.hamming(N1)
      X = np.zeros((1+nfft/2,numWindows))
      k = 0
      for i in n:
          if i+N1-1 > len(x):
              break
      
          x_seg  = x[i:i+N1]*W
          X_seg  = abs(fft(x_seg, nfft))
          X[:,k] = X_seg[0:1+nfft/2]
  
          k += 1
  
      return (X,nfft,numWindows)
  
  def filtbank(self,numFilts, minfrq, maxfrq, width, nfft):
      fftfrqs = np.arange(0,self.fs/2+self.fs/nfft,self.fs/nfft)
      #arange excludes stop point (like range), so we must use fs/2+fs/nfft instead of simply fs/2
  
      wts     = np.zeros((len(fftfrqs),numFilts));
      mb      = self.hz2mel(self.minfrq)
      mt      = self.hz2mel(self.maxfrq)
      melfrqs = np.linspace(mb,mt,self.numFilts+2);
      cntfrqs = np.zeros((self.numFilts+2));
  
      for k in range(0,numFilts+2):
          #note that range doesn't include the terminal point
          cntfrqs[k] = self.mel2hz(melfrqs[k])
  
      for k in range(0,self.numFilts):
          cfs = cntfrqs[k:k+3]; #doesn't take terminal point... weird
          cfs = cfs[1]+width*(cfs-cfs[1])
  
          loslope = (fftfrqs - cfs[0])/(cfs[1] - cfs[0])
          hislope = (cfs[2] - fftfrqs)/(cfs[2] - cfs[1])
  
          wts_temp = np.minimum(loslope,hislope)
          wts_temp = np.maximum(0,wts_temp)
  
          wts[:,k] = wts_temp
  
      return wts
  
  def dct(self,Q,numcep):
      #(the output of) this routine is essentially identical to MATLAB's
      S = Q.shape
      cos_arg = np.arange(1,2*S[0],2)
      dct_mat = np.zeros((S[0],S[0]))
      for k in np.arange(0,S[0]):
        dct_mat[k,:] = math.sqrt(2.0/S[0])*np.cos(math.pi*0.5*k*cos_arg/S[0])
  
      dct_mat[0,:] = dct_mat[0,:]/math.sqrt(2.0)
      
      C = np.dot(dct_mat,Q)
      C = C[0:self.numcepBasic,:]
  
      return C

  def idct(self,Q,numlen):
    S = Q.shape
    if numlen > S[0]:
      newQ = np.zeros((numlen,S[1]))
      newQ[0:S[0],:] = Q
      Q = newQ

    S = Q.shape
    cos_arg = np.arange(0,S[0])
    dct_mat = np.zeros((S[0],S[0]))
    for n in np.arange(0,S[0]):
      dct_mat[n,:] = math.sqrt(2.0/S[0])*np.cos(math.pi*0.5*cos_arg*(2*n+1)/S[0])

    dct_mat[:,0] = dct_mat[:,0]/math.sqrt(2.0)

    R = np.dot(dct_mat,Q)

    return R
  
  def cmn(self,C):
      m = np.mean(C,1)
      for i in range(0,self.numcepBasic):
          C[i,:] = C[i,:] - m[i]
  
      return C
  
  def deltas(self,c,w):
      S = c.shape
      d = np.zeros((S[0],S[1]))
      for n in range(0,S[1]):
          d[:,n] = c[:,(n+w/2) % S[1]]-c[:,n-w/2] #negative indices wrap around
      d = d/w
      return d

  def loadWAVfile(self, filename):
      w  = scipy.io.wavfile.read(filename)
      self.x  = w[1]
      self.fs = w[0]
      return self.x
    
  def calcMelFeatures(self, data):
      x = self.preemph(data,self.a)
      
#       fr, wordspower, wordszeros, wordsdetect, ITL ,ITU,  word_fr, word_y  = RecordModule.detectSingleWord(range(len(x)), x)
#       x = word_y
      self.fs = 44100.0
 
#       pylab.plot(range(len(x)) , x)
#       pylab.xlabel("x")
#       pylab.ylabel("y")
#       pylab.show()
#        
#        
      
      
      outTuple = self.stft(x,self.t1,self.t2,self.fs)
      
      X          = outTuple[0]
      nfft       = outTuple[1]
      numWindows = outTuple[2]
      
      wts = self.filtbank(self.numFilts, self.minfrq, self.maxfrq, 
            self.width, nfft)
      
      Xp  = pow(X,2)
      wts = pow(wts.transpose(),2)
      P   = np.dot(wts,Xp)
      
      Q = np.log(P);
      C = self.dct(Q,self.numcepBasic)
      
      C_cmn = self.cmn(C);
      R_cmn = self.idct(C_cmn,128) #second parameter is length of iDCT 
      
      d1 = self.deltas(C_cmn,self.del_w)
      d2 = self.deltas(d1,self.dbl_del_w)
      
      C_out = np.zeros((3*self.numcepBasic,numWindows))
      
      C_out[0:self.numcepBasic,:]             = C_cmn
      C_out[self.numcepBasic:2*self.numcepBasic]   = d1
      C_out[2*self.numcepBasic:3*self.numcepBasic] = d2
#       C_out[0,:] = 0
#       C_out = C_out[:,0:self.numcepsBands]
      

#     usrednienie wartosci spektrum dla danych wspolcz. w danm przedziale czasu
      sizeBand = int(len(C_out.T) / self.numcepsBands)
      if(sizeBand==0):
          sizeBand=1
      lpBand = 0
      C_out2 = [[0 for x in range(self.numcepsBands)] for y in range(self.numallceps)] 

      if(False):
        for i in range(self.numallceps):
            lpBand = 0
            for j in range(len(C_out.T)):
                C_out2[i][lpBand] += C_out[i][j]
                if (j%sizeBand == (sizeBand-1)):
                    C_out2[i][lpBand] = C_out2[i][lpBand]/sizeBand
#                   print("b:",lpBand)
                    if(lpBand<self.numcepsBands-1):
                        lpBand+=1

      if(True):
          for kk in range(self.numallceps):
            for ll in range(self.numcepsBands) :
               amplMax = max(C_out[kk][ll*sizeBand:ll*sizeBand+sizeBand])
               amplMin = min(C_out[kk][ll*sizeBand:ll*sizeBand+sizeBand])
               if amplMax > abs(amplMin) :
                    C_out2[kk][ll] = amplMax
               else :
                    C_out2[kk][ll] = amplMin
      
      return C_out2
  
  def calcMelVectFeatures(self, data):
      vect_of_mccf = np.zeros(len(data))
#       print(data.shape)
#       for i in range(len(data)): 
#           vect_of_mccf[i] =  abs(data[i][0])

      for i in range(len(data)): 
        vect_of_mccf[i] =  sum(abs(data[i])) #max(data[i]) # RecordModule.arithmeticMean(data[i]) #
#         amplMax = max(data[i])
#         amplMin = min(data[i])
#         if amplMax > abs(amplMin) :
#             vect_of_mccf[i] = amplMax
#         else :
#             vect_of_mccf[i] = amplMin
          
          
      vect_of_mccf[0] = 0      
      return vect_of_mccf    
  
  
    
  def plotSpectrogram(self, data, title):
          plt.title(title)
          plt.imshow(data, origin='lower')
          plt.show()
    
  def setNumFilts(self, num):
      self.numFilts = num

if __name__ == "__main__":
    MelFeat = MelFeatures()
    for i in range(10):
        filename = "learn_set//podglos//"+str(i+1)+".wav"    
        print("please speak a word into the microphone")
 
    #     RecordModule.record_to_file(filename)
        print("done - result written to ", filename)
          
          
    ###########################################    
          
        rawdata = MelFeat.loadWAVfile(filename)
         
        MFCC    = MelFeat.calcMelFeatures(rawdata)
#         MFCC_s  = MelFeat.calcMelVectFeatures(MFCC)
          
      
        MelFeat.plotSpectrogram(MFCC, str(i+1)+".wav"  )
        
#         pylab.plot(range(len(MFCC_s)) , MFCC_s,'b')
#         pylab.xlabel("x")
#         pylab.ylabel("y")
    pylab.show()
