import math
import random

import numpy as np
import torch
from dynamics import lgvDynamics2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#######################  Cal_memory capacity  #######################
#######################  Cal_memory capacity  #######################
#######################  Cal_memory capacity  #######################


g_list = np.arange(0.5, 9.25, 0.25)
tau_list = np.arange(1, 11, 1)

mm_list = []
repeat_num = 10
N = 1000
dt = 0.01
steps = 1000
for _ in range(repeat_num):
    m_list = []
    L_test_list = []
    g = 2
    for j in range(len(tau_list)):
        tau = tau_list[j]
        T = 0
        variance = g ** 2 / N
        std_dev = np.sqrt(variance)
        J = np.random.normal(loc=0, scale=std_dev, size=(N, N))
        np.fill_diagonal(J, 0)
        gamma = 0
        x = np.random.randn(N)
        X_list = []
        L_list = []
        Loss = 0.0
        label_list = []
        for t in range(3 * steps):
            s = np.sin(0.01 * 2 * np.pi * t)
            label = np.sin(0.01 * 2 * np.pi * (t + tau))
            label_list.append(torch.ones(1, dtype=torch.float64) * label)
            X_list.append(torch.from_numpy(x).to(torch.float64))
            x = lgvDynamics2(N, x, J, dt, gamma, s, T)

        X_matrix = torch.stack(X_list).T  # N*T
        S_list = [torch.ones(1, dtype=torch.float64) * np.sin(2 * np.pi * 0.01 * i) for i in range(3 * steps)]  # T*1

        S_matrix = torch.stack(S_list)
        label_matrix = torch.stack(label_list)  # T*1
        X_matrix = X_matrix.to(torch.float64)
        S_matrix = S_matrix.to(torch.float64)
        label_matrix = label_matrix.to(torch.float64)

        W = torch.matmul(torch.linalg.inv(torch.matmul(X_matrix, X_matrix.T)), torch.matmul(X_matrix, S_matrix))
        Z_hat = torch.matmul(W.T, X_matrix)

        m_tau = 1 - torch.mean((Z_hat - label_matrix.T) ** 2) / 0.5
        m_list.append(m_tau)

    mm_list.append(m_list)


def plt_m(mm_list):
    m_array = np.array(mm_list)
    mean_m = np.mean(m_array, axis=0)
    std_deviation = np.std(m_array, axis=0, ddof=1)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    tau = range(1, len(tau_list) + 1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(tau, mean_m, yerr=std_deviation,
                 fmt='-o',
                 ecolor='lightcoral',
                 elinewidth=2,
                 capsize=4,
                 capthick=2,
                 alpha=0.8,
                 label='10 times average')

    plt.xlabel('tau')
    plt.ylabel('memory')
    plt.legend()
    plt.tight_layout()
    plt.show()
