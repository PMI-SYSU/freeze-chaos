import math
import random

import numpy as np
import torch

import torch
from dynamics import lgvDynamics
import matplotlib
from torch import optim

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


g_list = [2]

T = 0
repeat_num = 1

sigma_e, sigma_i, f, miu_e, miu_i, alpha = 0.9, 0.9, 0.4, 0.6, -0.6, 0.8
N = 1000
dt = 0.01
steps = 10000
for _ in range(repeat_num):
    phi = torch.tanh

    for i in range(len(g_list)):
        g = g_list[i]
        J = (g / N ** 0.5) * torch.randn(N, N)
        # J = new_J(sigma_e, sigma_i, f, miu_e, miu_i, N, alpha)
        x = (torch.rand(size=(N,)) - 0.5) * 2
        # x1, x2, x3, x_at_switch1, x_at_switch2, x_list = lgvDynamics(N, x, J, T, dt, steps)
        x1, x2, x3, x4, x5, x6, x_list = lgvDynamics(N, x, J, T, dt, steps)


epochs = list(range(1, 3*steps+1))
epochs = [x / 100 for x in epochs]
plt.figure(figsize=(5, 4))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
cmap_blue = plt.cm.Blues
cmap_red = plt.cm.Reds
plt.plot(epochs, x1, color=cmap_blue(0.3), linewidth=2, label='x1')
plt.plot(epochs, x2, color=cmap_blue(0.5), linewidth=2, label='x1')
plt.plot(epochs, x3, color=cmap_blue(0.8), linewidth=2, label='x1')
plt.plot(epochs, x4, color=cmap_red(0.3), linewidth=2, label='x1')
plt.plot(epochs, x5, color=cmap_red(0.5), linewidth=2, label='x1')
plt.plot(epochs, x6, color=cmap_red(0.8), linewidth=2, label='x1')
plt.xlim(left=0)
plt.xlim(right=300)
plt.xticks([100, 200, 300])
# plt.axvline(x=steps*100, color='gray', linestyle='--')
# plt.axvline(x=2*steps*100, color='gray', linestyle='--')
# plt.title('chaos')
plt.xlabel('time')
plt.ylabel('Activity x')
# plt.legend()
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'dynamics_{timestamp}.pdf'
plt.savefig(filename, format='pdf')
plt.show()