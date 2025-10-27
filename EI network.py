import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


#######################  J_matrix  #######################
#######################  J_matrix  #######################

def new_J(sigma_e, sigma_i, f, miu_e, miu_i, N, alpha):
    sigma_e = sigma_e/np.sqrt(N)
    sigma_i = sigma_i / np.sqrt(N)
    miu_e = miu_e / np.sqrt(N)
    miu_i = miu_i / np.sqrt(N)
    S = torch.zeros((N,N))
    for i in range(N):
        for j in range(N):
            a = torch.rand(1).item()
            if a < alpha:
                S[i, j] = 1
    A = torch.randn(N, N)
    diagonal = torch.full((N,), sigma_i, dtype=torch.float)
    num_a = int(f * N)
    diagonal[:num_a] = sigma_e
    D = torch.diag(diagonal)
    u = torch.full((N, 1), 1.0, dtype=torch.float)
    column_vector = torch.full((N, 1), miu_i, dtype=torch.float)
    column_vector[:num_a] = miu_e
    AD = torch.matmul(A, D)
    M = torch.matmul(u, column_vector.T)
    W = S * (AD + M)
    # return W, S, A, D, M
    return W
