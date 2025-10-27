import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

N = 1000
def delta(i, j):
    """ delta f """
    if i == j:
        return 1
    else:
        return 0


def Jac_metrics(x, J):    # Vanilla RNN
    D = np.zeros((N, N))
    phiPrime = lambda x: 1 - np.tanh(x) ** 2
    for i in range(N):
        for j in range(N):
            D[i, j] = -delta(i, j) + J[i, j] * phiPrime(x)[j]
    return D


def Jac_metrics_2(x, J):  # construct RNN
    D = np.zeros((N, N))
    phi = np.tanh
    phiPrime = lambda x: 1 - np.tanh(x) ** 2
    phiPrime_prime = lambda x: (1 - np.tanh(x) ** 2) * (-2 * np.tanh(x))
    for i in range(N):
        for j in range(N):
            if i == j :
                D[i, j] = -1 - phiPrime_prime(x)[i]*np.sum(J[:, i] * (np.dot(J, x)
                                                                      -x))-phiPrime(x)[i] *phiPrime(x)[i] *np.sum(J[:, i] * J[:, i])
            else:
                D[i, j] = J[i, j] * phiPrime(x)[j] - phiPrime(x)[i] * (
                        np.sum(J[:, i] * J[:, j]) * phiPrime(x)[j] - J[j, i])
    return D


def get_eigen_value(D):
    eigenvalues = np.linalg.eigvals(D)
    real_part = np.real(eigenvalues)
    imag_part = np.imag(eigenvalues)
    return real_part, imag_part
