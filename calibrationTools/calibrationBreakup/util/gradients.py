import numpy as np
from scipy.spatial import cKDTree


def computeGrad(tree, Input, Output):
    dim = Input.shape[1]
    nSnapshots = Input.shape[0]
    gradient = np.zeros((nSnapshots, dim))
    A = np.zeros((dim + 1, dim + 1))
    for idim in range(dim + 1):
        A[0, idim] = 1

    for isnap in range(nSnapshots):
        xtarget = Input[isnap, :]
        dd, ii = tree.query(xtarget, k=dim + 1)

        # Compute gradient in each direction
        for jdim in range(dim):
            for idim in range(dim):
                A[idim + 1, jdim + 1] = Input[ii[jdim + 1], idim] - Input[ii[0], idim]
        det = np.linalg.det(A)
        # print(det)
        # if abs(det)<1e-20:
        #   #print(isnap)
        #   print(A[:,1])

        # iprint(A)
        for idim in range(dim):
            B = np.zeros((dim + 1, 1))
            B[idim + 1] = 1
            Alpha = np.linalg.lstsq(A, B, rcond=None)[0]
            gradient[isnap, idim] = 0
            for i in range(dim + 1):
                # print(ii[i])
                # print(Alpha[i])
                # print(Output[ii[i]])
                gradient[isnap, idim] = gradient[isnap, idim] + Alpha[i] * Output[ii[i]]

    return gradient
