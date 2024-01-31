import gradients as grad
import numpy as np
from scipy.spatial import cKDTree


def getActiveSubspaces(Input, Output):
    # Rescale input
    ndim = Input.shape[1]
    nSnapshots = Input.shape[0]

    rescale1 = np.zeros(ndim)
    rescale2 = np.zeros(ndim)

    for idim in range(ndim):
        rescale1[idim] = np.amax(Input[:, idim]) - np.amin(Input[:, idim])
        rescale2[idim] = np.amin(Input[:, idim])

    for idim in range(ndim):
        Input[:, idim] = (Input[:, idim] - rescale2[idim]) / rescale1[idim]

    # Create a tree for finding nearest neighbours
    tree = cKDTree(Input)
    gradient = grad.computeGrad(tree, Input, Output)

    # Do the active subspace part
    C = np.zeros((ndim, ndim))
    line = np.zeros((1, ndim))
    column = np.zeros((ndim, 1))

    for isnap in range(nSnapshots):
        line[0, :] = gradient[isnap, :]
        column[:, 0] = gradient[isnap, :]
        C = C + np.matmul(column, line)

    C = C / nSnapshots

    # Eigen decomposition of C : C = W Lambda W^T
    Lambda, W = np.linalg.eig(C)
    W = np.transpose(W)

    LambdaTmp = Lambda.copy()

    maxLambda1 = 0

    for i in range(len(LambdaTmp)):
        if LambdaTmp[i] > maxLambda1:
            maxLambda1 = LambdaTmp[i]
            maxW1 = W[i]
            indMax = i

    # Project on the most active subspace
    InputProj = np.zeros(nSnapshots)
    for isnap in range(nSnapshots):
        for idim in range(ndim):
            InputProj[isnap] = InputProj[isnap] + Input[isnap, idim] * maxW1[idim]

    # Rescale input
    for idim in range(ndim):
        Input[:, idim] = Input[:, idim] * rescale1[idim] + rescale2[idim]

    return maxW1, InputProj


def projInput(Input, direction, rescale1, rescale2):
    newInput = Input.copy()

    # Rescale input
    ndim = newInput.shape[1]
    nSnapshots = newInput.shape[0]

    for idim in range(ndim):
        newInput[:, idim] = (newInput[:, idim] - rescale2[idim]) / rescale1[idim]

    # Project on the most active subspace
    InputProj = np.zeros(nSnapshots)
    for isnap in range(nSnapshots):
        for idim in range(ndim):
            InputProj[isnap] = (
                InputProj[isnap] + newInput[isnap, idim] * direction[idim]
            )

    return InputProj
