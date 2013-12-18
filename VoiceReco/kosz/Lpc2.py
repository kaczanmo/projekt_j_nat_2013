#!/usr/bin/env python
  
from rlda import min_rlda, rlda
from Numeric import *
from MLab import *
from rawdata import PhonemeData
from lpc import LPC, lsf
  
def main():
  
    rd = PhonemeData('../data/phonemes')
    s_1 = rd.loadPhoneme('o')
    s_2 = rd.loadPhoneme('u')
  
    f_1 = []
    f_2 = []
  
    print 'Calculating LSP for 1st set'
    for idx, i in enumerate(s_1):
        f = LPC(i[:2048], 256)
  
        a, b = lsf(f, angles=True, FS=44100.)
        x = []
        for i in range(len(a)):
            x.append((a[i]+b[i])/2)
  
        f_1.append(x)
  
    print 'Calculating LSP for 2nd set'
    for idx, i in enumerate(s_2):
        f = LPC(i[:2048], 256)
  
        a, b = lsf(f, angles=True, FS=44100.)
        x = []
        for i in range(len(a)):
            x.append((a[i]+b[i])/2)
  
        f_2.append(x)
  
    f_1l = f_1[:len(f_1)*3/4]
    f_2l = f_2[:len(f_2)*3/4]
    f_1t = f_1[len(f_1)*3/4:]
    f_2t = f_2[len(f_2)*3/4:]
  
    print 'Searching for min'
    lam = min_rlda(array(f_1l), array(f_2l))
  
    p_x, p_0 = rlda(array(f_1l), array(f_2l), lam)
  
    print 'Checking'
    rigth = 0
    for i in f_1t:
        g_rlda1 = matrixmultiply(p_x, transpose(i)) + p_0
        if g_rlda1 > 0:
            rigth+=1
  
    for i in f_2t:
        g_rlda1 = matrixmultiply(p_x, transpose(i)) + p_0
        if g_rlda1 < 0:
            rigth+=1
  
    print 'Right:', float(rigth)/(len(f_1t)+len(f_2t))
  
  
if __name__ == '__main__':
    main()