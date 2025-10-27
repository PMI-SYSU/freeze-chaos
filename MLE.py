import math
import random

import numpy as np
import torch
from dynamics import lgvDynamics2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#############   orbit separation method  ################
#############   orbit separation method  ################
#############   orbit separation method  ################

def run_experiment():
    N = 1000
    dt = 0.01
    Time = 50.0
    steps = int(Time / dt)

    size = 10
    g_list = np.linspace(0.2, 5, size)

    LYP = []
    for g in g_list:
        lambda_list = []

        X = np.random.randn(N)
        a = np.random.randn(N)
        epsilon = 1e-5 * a / np.linalg.norm(a)
        X_1 = X + epsilon

        variance = g ** 2 / N
        std_dev = np.sqrt(variance)

        J = np.random.normal(loc=0, scale=std_dev, size=(N, N))
        np.fill_diagonal(J, 0)

        for t in range(steps):
            X = lgvDynamics2(N, X, J, dt, 1, 0, 0)
            X_1 = lgvDynamics2(N, X_1, J, dt, 1, 0, 0)
            lambda_list.append(np.log(np.linalg.norm(X - X_1) / 1e-5))
            X_1 = X + 1e-5 * (X - X_1) / np.linalg.norm(X - X_1)

        lyp = np.mean(lambda_list[-8000:])
        LYP.append(lyp / dt)

    return g_list, LYP

g_list, LYP = run_experiment()

def plt_LYP(g_list, LYP):
    plt.figure(figsize=(10, 6))
    plt.plot(g_list, LYP, marker='o', linewidth=2, markersize=4)

    plt.xlabel('g', fontsize=12)
    plt.ylabel('LYP', fontsize=12)

    plt.tight_layout()
    plt.show()
