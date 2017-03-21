"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To show how TDA can be used to quantify quasiperiodicity
in an audio clip of horse whinnies
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.interpolate as interp

from TDA import *
from SlidingWindow import *
import scipy.io.wavfile

if __name__ == '__main__':
    #Read in the audio file.  Fs is the sample rate, and
    #X is the audio signal
    Fs, X = scipy.io.wavfile.read("horsewhinnie.wav")
    
    #These variables are used to adjust the window size
    F0 = 493 #First fundamental frequency
    G0 = 1433 #Second fundamental frequency

    ###TODO: Modify this variable (time in seconds)
    time = 0.85

    #Step 1: Extract an audio snippet starting at the chosen time
    SigLen = 512 #The number of samples to take after the start time
    iStart = int(round(time*Fs))
    x = X[iStart:iStart + SigLen]
    W = int(round(Fs/G0))

    #Step 2: Get the sliding window embedding
    Y = getSlidingWindow(x, W, 2, 2)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    #Step 3: Do the 1D rips filtration
    PDs = doRipsFiltration(Y, 1)
    PD = PDs[1]

    #Step 4: Figure out the second largest persistence
    sP = 0
    sPIdx = 0
    if PD.shape[0] > 1:
        Pers = PD[:, 1] - PD[:, 0]
        sPIdx = np.argsort(-Pers)[1]
        sP = Pers[sPIdx]
        
    #Step 5: Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Starting At %g Seconds"%time)
    plt.plot(time + np.arange(SigLen)/float(Fs), x)
    plt.xlabel("Time")
    plt.subplot(122)
    plotDGM(PD)
    plt.hold(True)
    plt.plot([PD[sPIdx, 0]]*2, PD[sPIdx, :], 'r')
    plt.scatter(PD[sPIdx, 0], PD[sPIdx, 1], 20, 'r')
    plt.title("Second Largest Persistence: %g"%sP)
    
    plt.show()
