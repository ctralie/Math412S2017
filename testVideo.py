"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To show how to do a normalized sliding window embedding of a 
video of a man waving his hands.  Then showing how TDA picks up on this
as a persistent 1-cycle
"""
from VideoTools import *
from TDA import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #Load in video
    (XOrig, IDims) = loadVideo("HandClapping.mp4")
    
    #Reduce dimension losslessly by doing an SVD
    X = getPCAVideo(XOrig)
    
    #Cut down on drift with a gradient in time
    DerivWin = 5
    [X, validIdx] = getTimeDerivative(X, DerivWin)
    
    #Do sliding window
    dim = 70
    Tau = 0.5
    dT = 1
    XS = getSlidingWindowVideo(X, dim, Tau, dT)
    
    #Z-normalize sliding window embedding
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    
    #Do PCA
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(XS)
    eigs = pca.explained_variance_ratio_
    
    #Do TDA
    PDs = doRipsFiltration(XS, 1, coeff=2)
    
    #Plot Results
    plt.figure(figsize=(12, 5))
    #Plot PCA
    plt.subplot(121)
    plt.scatter(Y[:, 0], Y[:, 1], 20, np.arange(Y.shape[0]))
    plt.title('PCA, Variance Explained: %g %s'%(np.sum(eigs), '%'))
    #Plot the 1D persistence diagram
    plt.subplot(122)
    plotDGM(PDs[1])
    plt.show()
