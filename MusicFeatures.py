"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: Code to compute features on audio files, including MFCC
and audio novelty
"""
import numpy as np
import numpy.linalg as linalg
import scipy
from scipy.io import wavfile
from scipy.io import savemat
from scipy.fftpack import dct
import matplotlib.pyplot as plt


#Mirror what Matlab's code does
def STFTNoOverlapZeropad(X, hopSize):
    N = X.shape[0]
    ham = np.hamming(hopSize) #Use hamming window
    #Zeropad X so that there are an integer number of hopSize intervals
    NWindows = int(np.ceil(N/float(hopSize)))
    S = np.zeros((hopSize, NWindows), dtype = np.float32)
    x = np.zeros(hopSize)
    for i in range(NWindows):
        x = 0*x
        n = len(X[i*hopSize:(i+1)*hopSize])
        x[0:n] = X[i*hopSize:(i+1)*hopSize]
        S[:, i] = np.abs(np.fft.fft(ham*x))
    return S

def getMFCC(X, Fs, hopSize, MEL_NBANDS = 40, MEL_NMFCC = 20, MEL_MINFREQ = 50, MEL_MAXFREQ = 4000, lifterexp = 0):
    """
    Return the MFCC coefficients
    :param X: Audio Signal
    :param Fs: Sample Rate
    :param hopSize: Hop size between windows
    :param lifterexp: Liftering to emphasize higher coefficients
    """
    X = np.array(X, dtype=np.float32)
    #Data is not normalized when read in.  Assume 16 bit
    X = X/(2.0**15)
    if len(X.shape) > 1 and X.shape[1] > 1:
        #Merge to mono if there is more than one channel
        X = X.mean(1)
    X = X.flatten()

    #Compute spectrogram
    S = STFTNoOverlapZeropad(X, hopSize) #Spectrogram
    SHalf = S[0:hopSize/2+1, :] #Non-redundant spectrogram
    P = SHalf**2 #Periodogram
    NSpectrumSamples = SHalf.shape[0]
    NAWindows = S.shape[1]

    #Step 1: Warp to the mel-frequency scale
    melbounds = np.array([MEL_MINFREQ, MEL_MAXFREQ])
    melbounds = 1125*np.log(1 + melbounds/700.0)
    mel = np.linspace(melbounds[0], melbounds[1], MEL_NBANDS)
    binfreqs = 700*(np.exp(mel/1125.0) - 1)
    binbins = np.floor(((hopSize-1)/float(Fs))*binfreqs) #Floor to the nearest bin
    binbins = np.array(binbins, dtype=np.int64)

    #Step 2: Create mel triangular filterbank
    melfbank = np.zeros((MEL_NBANDS, NSpectrumSamples))
    for i in range(MEL_NBANDS):
       thisbin = binbins[i]
       lbin = thisbin
       if i > 0:
           lbin = binbins[i-1]
       rbin = thisbin + (thisbin - lbin)
       if i < MEL_NBANDS - 1:
           rbin = binbins[i+1]
       melfbank[i, lbin:thisbin+1] = np.linspace(0, 1, 1 + (thisbin - lbin))
       melfbank[i, thisbin:rbin+1] = np.linspace(1, 0, 1 + (rbin - thisbin))

    #Step 3: Apply mel filterbank to periodogram, and compute log of the result
    MFCC = np.array( [melfbank.dot(P[:, i].T) for i in range(NAWindows)] ).T
    MFCC[MFCC <= 0] = 1
    MFCC = np.log(MFCC)

    #Step 4: Compute DCT and return mel coefficients
    MFCCDCT = dct(MFCC, axis = 0, norm = 'ortho')
    MFCCDCT = MFCC[0:MEL_NMFCC, :]
    
    #Step 5: Do Liftering
    coeffs = np.arange(MFCCDCT.shape[0])**lifterexp
    coeffs[0] = 1
    MFCCDCT = coeffs[:, None]*MFCCDCT
    
    return (MFCC, MFCCDCT, P)

def getAudioNovelty(X, Fs, hopSize):
    (MFCC, MFCCDCT, P) = getMFCC(X, Fs, hopSize)
    diff = MFCC[:, 1::] - MFCC[:, 0:-1]
    diff[diff < 0] = 0
    return np.sum(diff, 0)

if __name__ == '__main__':
    Fs, X = scipy.io.wavfile.read("journey.wav")
    hopSize = 512
    (MFCC, MFCCDCT, P) = getMFCC(X, Fs, hopSize)
    novFn = getAudioNovelty(X, Fs, hopSize)
    
    nsamples = 400
    P = P[:, 0:nsamples]
    novFn = novFn[0:nsamples]
    t = np.arange(nsamples)*hopSize/float(Fs)
    
    plt.subplot(211)
    plt.imshow(np.log(P), cmap = 'afmhot', aspect = 'auto')
    plt.title("Spectrogram")
    plt.axis('off')
    plt.subplot(212)
    plt.plot(t, novFn)
    plt.title("Audio Novelty Function")
    plt.xlabel("Time (Sec)")
    plt.xlim([0, np.max(t)])
    plt.show()
