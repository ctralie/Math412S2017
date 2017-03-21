"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To show how TDA can be used to quantify how periodic
an audio clip is.  Simple example with music versus speech.
Show how doing a delay embedding on raw audio is a bad idea when
the length of the period is on the order of seconds, and how
"audio novelty functions" come in handy
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.interpolate as interp

from TDA import *
from SlidingWindow import *
from MusicFeatures import *
import scipy.io.wavfile

if __name__ == '__main__':
    #Don't Stop Believing
    FsMusic, XMusic = scipy.io.wavfile.read("journey.wav") 
    FsSpeech, XSpeech = scipy.io.wavfile.read("speech.wav")
    
    #Step 1: Try a raw delay embedding
    #Note that dim*Tau here spans a half a second of audio, 
    #since Fs is the sample rate
    dim = round(FsMusic/200)
    Tau = 100
    dT = FsMusic/100
    Y = getSlidingWindowInteger(XMusic[0:FsMusic*3], dim, Tau, dT)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    PDs = doRipsFiltration(Y, 1)
    pca = PCA()
    Z = pca.fit_transform(Y)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("2D PCA Raw Audio Embedding")
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.subplot(122)
    plotDGM(PDs[1])
    plt.title("Persistence Diagram")
    
    
    #Step 2: Do sliding window on audio novelty functions
    #(sliding window of sliding windows!)
    hopSize = 512
    
    #First do audio novelty function on music
    novFnMusic = getAudioNovelty(XMusic, FsMusic, hopSize)
    dim = 20
    #Make sure the window size is half of a second, noting that
    #the audio novelty function has been downsampled by a "hopSize" factor
    Tau = (FsMusic/2)/(float(hopSize)*dim)
    dT = 1
    Y = getSlidingWindowInteger(novFnMusic, dim, Tau, dT)
    print("Y.shape = ", Y.shape)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    PDs = doRipsFiltration(Y, 1)
    pca = PCA()
    Z = pca.fit_transform(Y)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("2D PCA Music Novelty Function Sliding Window")
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.subplot(122)
    plotDGM(PDs[1])
    plt.title("Persistence Diagram")

    
    #Now do audio novelty function on speech
    novFnSpeech = getAudioNovelty(XSpeech, FsSpeech, hopSize)
    dim = 20
    #Make sure the window size is half of a second, noting that
    #the audio novelty function has been downsampled by a "hopSize" factor
    Tau = (FsSpeech/2)/(float(hopSize)*dim)
    dT = 1
    Y = getSlidingWindowInteger(novFnSpeech, dim, Tau, dT)
    print("Y.shape = ", Y.shape)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    PDs = doRipsFiltration(Y, 1)
    pca = PCA()
    Z = pca.fit_transform(Y)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("2D PCA Speech Novelty Function Sliding Window")
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.subplot(122)
    plotDGM(PDs[1])
    plt.title("Persistence Diagram")
    plt.show()
    
