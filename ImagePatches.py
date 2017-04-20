import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from TDA import *

def getPatches(I, dim):
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
    return P

def getCirclePatches(N, dim):
    R = N/2
    [I, J] = np.meshgrid(np.arange(N) ,np.arange(N))
    Im = ((I-R)**2 + (J-R)**2) < (0.5*R*R)
    Im = 1.0*Im
    P = getPatches(Im, dim)
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

def getGradientPatches(dim, NPatches):
    print "TODO"

if __name__ == '__main__':
    (Im, P) = getCirclePatches(40, 5)
    sio.savemat("P.mat", {"P":P})
    PDs = doRipsFiltration(P, 2)
    print PDs[2]
    H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.show()
