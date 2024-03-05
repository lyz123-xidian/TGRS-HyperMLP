import torch
import numpy as np
from torch import nn
import math
from einops.layers.torch import Rearrange
from einops import rearrange
# from rpca import robust_pca
def shrink(X,tau):
    V = torch.clone(X).reshape((X.numel(),))
    for i in range(V.numel()):
        t = torch.abs(V[i]) - tau
        a = 0
        a = np.array(a)
        a = torch.from_numpy(a)
        a = torch.maximum(t, a)
        # a = torch.max(t, 0)[0]
        V[i] = torch.copysign(a, V[i])
        # if V[i] == -0:
        #     V[i] = 0

    c = X.shape
    pp = V.reshape(X.shape)
    return pp
def svd_shrink(X, tau):
    U,s,V = torch.linalg.svd(X,full_matrices=False)
    a1 = torch.diag(shrink(s, tau))
    # b = a1 / V
    b = torch.mm(a1, V)
    a = torch.mm(U, b)

    return a

def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    a = X.numel()
    V = torch.reshape(X, (X.numel(),))
    for i in range(V.numel()):
        accum += torch.abs(V[i] ** 2)
    return torch.sqrt(accum)

def converged(M, L, S):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    # print("error =", error)
    return error <= 10e-6

def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return torch.max(torch.sum(X, dim=0))

def robust_pca(M):
    """
    Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """
    L = torch.zeros(M.shape)
    S = torch.zeros(M.shape)
    Y = torch.zeros(M.shape)
    # print(M.shape)
    mu = torch.div(M.shape[0] * M.shape[1], 4.0 * L1Norm(M))
    # mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    tt = mu * lamb
    while not converged(M, L, S):
        L = svd_shrink(M - S - (mu ** -1) * Y, mu)
        S = shrink(M - L + (mu ** -1) * Y, tt)
        Y = Y + mu * (M - L - S)
    return L, S

if __name__ == "__main__":
    img = torch.ones((121,128))
    output = robust_pca(img)[0]
    print("Shape of out :", output.shape)