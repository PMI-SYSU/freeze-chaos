import math
import random
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib
from torch import optim
matplotlib.use('TkAgg')  # 可以尝试不同的后端，如 'TkAgg', 'Qt5Agg', 'WebAgg', 等。
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics



N = 1000
def calculate_skewness_kurtosis(data):
    n = len(data)
    if n < 3:
        raise ValueError
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)

    if stdev == 0:
        raise ValueError

    m3 = sum((x - mean) ** 3 for x in data) / n
    m4 = sum((x - mean) ** 4 for x in data) / n

    skewness = (m3 / (stdev ** 3)) * (n * (n - 1)) ** 0.5 / (n - 2)
    kurtosis = (m4 / (stdev ** 4)) - 3
    return (skewness, kurtosis)

def new_J(sigma_e, sigma_i, f, miu_e, miu_i, N, alpha):
    sigma_e = sigma_e/np.sqrt(N)
    sigma_i = sigma_i / np.sqrt(N)
    miu_e = miu_e / np.sqrt(N)
    miu_i = miu_i / np.sqrt(N)
    # S = torch.randint(0, 2, (N, N), dtype=torch.float)  # 将S的数据类型设置为float\
    S = torch.zeros((N,N))
    for i in range(N):
        for j in range(N):
            a = torch.rand(1).item()
            if a < alpha:
                S[i, j] = 1
    A = torch.randn(N, N)
    diagonal = torch.full((N,), sigma_i, dtype=torch.float)  # 设置为float
    num_a = int(f * N)
    diagonal[:num_a] = sigma_e
    D = torch.diag(diagonal)
    u = torch.full((N, 1), 1.0, dtype=torch.float)  # 设置为float
    column_vector = torch.full((N, 1), miu_i, dtype=torch.float)  # 设置为float
    column_vector[:num_a] = miu_e
    AD = torch.matmul(A, D)
    M = torch.matmul(u, column_vector.T)
    W = S * (AD + M)
    # return W, S, A, D, M
    return W
# sigma_e, sigma_i, f, miu_e, miu_i, alpha = 0.3, 0.3, 0.4, 0.6, -0.4, 0.2
# J = new_J(sigma_e, sigma_i, f, miu_e, miu_i, N, alpha)

def lgvDynamics(N, x, J, T, dt, steps, I, r_max, gamma):  # gamma == 1
    phi = lambda x: r_max/(1 + torch.exp(-x))
    phiPrime = lambda x: r_max * (1/(1 + torch.exp(-x))) * (1 - 1/(1 + torch.exp(-x)))
    # phi = torch.tanh
    # phiPrime = lambda x: 1 - torch.tanh(x) ** 2
    x_list = []
    x = x.float()
    J = J.float()
    for i in range(int(steps)):
        x = x - dt * torch.mv(-torch.eye(N) + gamma * (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
                2 * T * dt) * torch.randn(size=(N,)) + I[:, i]
        x_list.append(x)
    return x_list

def LgvDynamics(N, x, J, T, dt, steps, r_max,gamma):  # gamma == 0
    phi = lambda x: r_max/(1 + torch.exp(-x))
    phiPrime = lambda x: r_max * (1/(1 + torch.exp(-x))) * (1 - 1/(1 + torch.exp(-x)))
    # phi = torch.tanh
    # phiPrime = lambda x: 1 - torch.tanh(x) ** 2
    x_list = []
    x = x.float()
    J = J.float()
    for i in range(int(steps)):
        x = x - dt * torch.mv(-torch.eye(N) + gamma * (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
                2 * T * dt) * torch.randn(size=(N,))
        x_list.append(x)
    return x

def get_I(t, dt, f, N, I0):   # stimulate matrix
    z = torch.zeros((N, int(100 * t)))
    for m in range(N):
        I_list = []
        for i in range(int(t)):
            u = torch.ones(int(1 / dt)) * I0  # dt==0.01
            v = torch.zeros(100)
            indices = torch.randperm(100)[:int(f)]
            v[indices] = 1
            I = u * v
            I_list.append(I)
        I = torch.cat(I_list, dim=0)
        z[m, :] = I
    return z
#
repeat_num = 5
all_results = []
for h in range(repeat_num):
    I_all = torch.ones((N, 1)) * 0.001
    f_list = [i for i in range(1, 101, 1)]
    NRL = []
    for f in f_list:
        max_t = 2000
        r_max = 0.2
        dt = 0.01
        t = 1
        I0 = 0.001
        T = 0
        m = 0
        steps = t * 100  # 给t秒刺激
        gamma = 0
        I = get_I(t, dt, f, N, I0)
        sigma_e, sigma_i, f, miu_e, miu_i, alpha = 0.3, 0.3, 0.4, 0.6, -0.4, 0.2
        J = new_J(sigma_e, sigma_i, f, miu_e, miu_i, N, alpha)
        x = (torch.rand(size=(N,)) - 0.5) * 2  #
        x_list = lgvDynamics(N, x, J, T, dt, steps, I, r_max, gamma)  # 前steps步给刺激
        # r_list = [1/(1 + torch.exp(-x[0])) for x in x_list]
        # print(r_list)
        x = x_list[-1]
        x_list = lgvDynamics(N, x, J, T, dt, 1, I_all, r_max, gamma)  # steps+1步同时给刺激
        x = x_list[-1]
        nrl = [0 for num in range(N)]
        while 0 in nrl and m < max_t:
            x = LgvDynamics(N, x, J, T, dt, 1, r_max, gamma)

            m += 1
            r_x = r_max / (1 + torch.exp(-x))
            # if m == 1:
            #     # print(r_x)
            #     print(x)
            # print(r_x[0])
            sample = torch.rand(N)
            a = r_x - sample
            b = torch.nonzero(a > 0, as_tuple=False)
            if b.numel() != 0:
                for k in range(len(b)):
                    if nrl[b[k]] == 0:
                        nrl[b[k]] = m
        NRL.append(nrl)
    midd = [statistics.median(data_list) for data_list in NRL]
    all_results.append(midd)
    print(h)


# # all_nrl = np.array([NRL1[:100], NRL2[:100], NRL3[:100], NRL4[:100], NRL5[:100]])
# all_nrl = np.array(NRL)
#
# mean_nrl = np.mean(all_nrl, axis=1)
# std_nrl = np.std(all_nrl, axis=1)
#
#
# plt.figure(figsize=(10, 8))
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
# plt.plot(f_list, mean_nrl, color='blue', linewidth=2, label='Mean NRL')
# plt.fill_between(f_list, mean_nrl - std_nrl, mean_nrl + std_nrl,
#                  color='lightblue', alpha=0.5, label='±1 SD')
#
# plt.xlabel('Stimulus Frequency f', fontsize=25)
# plt.ylabel(r'NRL ($\times$ 10ms)', fontsize=25)
#
#
# plt.tight_layout()
# from datetime import datetime
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = f'dynamics_{timestamp}.pdf'
# plt.savefig(filename, format='pdf')
# plt.show()

