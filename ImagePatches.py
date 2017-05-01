"""
Programmer: Chris Tralie
Purpose: To explore the space of natural image patches using TDA tools
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from TDA import *

def plotPatches(P):
    N = P.shape[0]
    d = np.sqrt(P.shape[1])
    dgrid = int(np.ceil(np.sqrt(N)))
    for i in range(N):
        plt.subplot(dgrid, dgrid, i+1)
        I = np.reshape(P[i, :], [d, d])
        plt.imshow(I, interpolation = 'nearest', cmap = 'gray')
        plt.axis('off')

def getPatches(I, dim, doPlotPatches = False):
    """
    Given an image I, return all of the dim x dim patches in I
    :param I: An M x N image
    :param d: The dimension of the square patches
    :returns P: An (M-d+1)x(N-d+1)x(d^2) array of all patches
    """
    #http://stackoverflow.com/questions/13682604/slicing-a-numpy-image-array-into-blocks
    shape = np.array(I.shape*2)
    strides = np.array(I.strides*2)
    W = np.asarray(dim)
    shape[I.ndim:] = W
    shape[:I.ndim] -= W - 1
    if np.any(shape < 1):
        raise ValueError('Window size %i is too large for image'%dim)
    P = np.lib.stride_tricks.as_strided(I, shape=shape, strides=strides)
    P = np.reshape(P, [P.shape[0], P.shape[1], dim*dim])
    if doPlotPatches:
        plotPatches(np.reshape(P, [P.shape[0]*P.shape[1], dim*dim]))
        plt.savefig("CirclePatches.svg", bbox_inches='tight')
    return P

def getCirclePatches(N, dim, doPlotPatches = False):
    R = N/2
    [I, J] = np.meshgrid(np.arange(N) ,np.arange(N))
    Im = ((I-R)**2 + (J-R)**2) < (0.5*R*R)
    Im = 1.0*Im
    P = getPatches(Im, dim, doPlotPatches)
    P = np.reshape(P, (P.shape[0]*P.shape[1], dim*dim))

    #Remove redundant patches to cut down on computation time
    toKeep = [0]
    XSqr = np.sum(P**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*P.dot(P.T)
    for i in range(1, D.shape[0]):
        if np.sum(D[i, 0:i] == 0) > 0:
            continue
        toKeep.append(i)
    P = P[np.array(toKeep), :]
    print "%i Circle Patches"%P.shape[0]
    return (Im, P)

def plotCirclePatches():
    (Im, P) = getCirclePatches(40, 5)
    plt.clf()
    sio.savemat("PCircle.mat", {"P":P})
    PDs = doRipsFiltration(P, 2)
    print PDs[2]
    H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.show()

def getGradientPatches(dim, NAngles, NSmooths):
    N = NAngles*NSmooths
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    phis = np.linspace(0, 2*np.pi, NSmooths+1)[0:NAngles]
    idx = 0
    [I, J] = np.meshgrid(np.arange(dim)*1.0, np.arange(dim)*1.0)
    I += 1.0
    J += 1.0
    for i in range(NAngles):
        a = np.cos(thetas[i])
        b = np.sin(thetas[i])
        for j in range(NSmooths):
            c = np.cos(phis[j])
            d = np.sin(phis[j])
            X = -1 + (2*I-1)/float(dim)
            Y = 1 - (2*J-1)/float(dim)
            patch = c*(a*X + b*Y)/2 + d*np.sqrt(3)*((a*X + b*Y)**2)/4
            P[idx, :] = patch.flatten()
            idx += 1
            # plt.imshow(patch, cmap = 'afmhot', interpolation = 'none')
            # plt.colorbar()
            # plt.show()
    return P

def getLinePatches(dim, NAngles, NOffsets, sigma = 1):
    N = NAngles*NOffsets
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    ps = np.linspace(-0.5*np.sqrt(2), 0.5*np.sqrt(2), NOffsets)
    idx = 0
    [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
    for i in range(NAngles):
        c = np.cos(thetas[i])
        s = np.sin(thetas[i])
        for j in range(NOffsets):
            patch = X*c + Y*s + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            P[idx, :] = patch.flatten()
            idx += 1
    return P

def plotLinePatches(P, name):
    plotPatches(P)
    plt.savefig("%sPatches.svg"%name, bbox_inches='tight')
    plt.clf()
    sio.savemat("P%s.mat"%name, {"P":P})

    plt.subplot(121)
    PDs = doRipsFiltration(P, 2, coeff = 2)
    print PDs[2]
    H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.title("$\mathbb{Z}2$ Coefficients")

    plt.subplot(122)
    PDs = doRipsFiltration(P, 2, coeff = 3)
    print PDs[2]
    H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.title("$\mathbb{Z}3$ Coefficients")
    plt.show()

def doLinePatchesVariation(dim, pres = 20):
    CUTOFF = 0.99
    pad = 3
    sigmas = np.linspace(0, 1, 101)[1::]
    H12s = np.zeros(len(sigmas))
    H22s = np.zeros(len(sigmas))
    H13s = np.zeros(len(sigmas))
    H23s = np.zeros(len(sigmas))
    PCAEigs = np.zeros(len(sigmas))
    pca = PCA()
    plt.figure(figsize=(18, 6))
    for i in range(len(sigmas)):
        plt.clf()
        print "Doing sigma = ", sigmas[i]
        P = getLinePatches(dim, pres, pres, sigmas[i])
        #First plot patches
        for j in range(pres):
            for k in range(pres):
                plt.subplot2grid((pres+pad, (pres+pad)*3), (j, k))
                I = np.reshape(P[j*pres+k, :], [dim, dim])
                plt.imshow(I, interpolation = 'nearest', cmap = 'gray')
                plt.axis('off')
        
        PDs = doRipsFiltration(P, 2, coeff = 2)
        H1 = PDs[1]
        H2 = PDs[2]
        if H1.size > 0:
            H12s[i] = np.max(H1[:, 1] - H1[:, 0])
        if H2.size > 0:
            H22s[i] = np.max(H2[:, 1] - H2[:, 0])

        plt.subplot2grid((pres+pad, (pres+pad)*3), (0, pres+pad), colspan=pres, rowspan=pres)
        H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
        plt.hold(True)
        H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
        plt.xlim([0, 30])
        plt.ylim([0, 30])
        plt.title("$\mathbb{Z}2$ Coefficients")

        PDs = doRipsFiltration(P, 2, coeff = 3)
        H1 = PDs[1]
        H2 = PDs[2]
        if H1.size > 0:
            H13s[i] = np.max(H1[:, 1] - H1[:, 0])
        if H2.size > 0:
            H23s[i] = np.max(H2[:, 1] - H2[:, 0])
        plt.subplot2grid((pres+pad, (pres+pad)*3), (0, 2*(pres+pad)), colspan=pres, rowspan=pres)
        H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
        plt.hold(True)
        H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
        plt.xlim([0, 30])
        plt.ylim([0, 30])
        plt.title("$\mathbb{Z}3$ Coefficients")
            
        Z = pca.fit_transform(P)
        x = np.cumsum(pca.explained_variance_ratio_)
        idx = np.where(x > CUTOFF)[0][0]
        PCAEigs[i] = float(idx)/min(P.shape[0], P.shape[1])
        
        plt.subplot2grid((pres+pad, (pres+pad)*3), (0, 0), colspan=pres, rowspan=2)
        plt.title("Sigma = %g\nDim = %i (Ratio = %.3g)"%(sigmas[i], idx, PCAEigs[i]))
        plt.axis('off')
        plt.savefig("%i.png"%i, bbox_inches = 'tight')
        
        sio.savemat("LinePatches.mat", {"sigmas":sigmas, "H12s":H12s, "H22s":H22s, "H13s":H13s, "H23s":H23s, "PCAEigs":PCAEigs})

if __name__ == '__main__':
    doLinePatchesVariation(50)
    P = getLinePatches(50, 16, 16, 0.275)
    plotLinePatches(P, "Line")
