import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


#######################  dynamics  #######################
#######################  dynamics  #######################
def lgvDynamics(N, x, J, T, dt, steps):  # gamma == 0
    phi = torch.tanh
    phiPrime = lambda x: 1 - torch.tanh(x) ** 2
    x_list = []
    x = x.float()
    J = J.float()
    for i in range(int(3*steps)):
        x_list.append(x)
        # print(i)
        if i < int(steps):
            x = x + dt * (-x + torch.matmul(J, phi(x))) # + math.sqrt(2 * T * dt) * torch.randn(size=(N,))
            # x = x - dt * torch.mv(-torch.eye(N) + (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
            #     2 * T * dt) * torch.randn(size=(N,))
        elif i == int(steps):
            x_at_switch1 = x
            # x = x + dt * (-x + torch.matmul(J, phi(x))) + math.sqrt(2 * T * dt) * torch.randn(size=(N,))
            x = x - dt * torch.mv(-torch.eye(N) + (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
                2 * T * dt) * torch.randn(size=(N,))
        elif int(steps) < i < int(2*steps):
            # x = x + dt * (-x + torch.matmul(J, phi(x))) + math.sqrt(2 * T * dt) * torch.randn(size=(N,))
            x = x - dt * torch.mv(-torch.eye(N) + (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
                2 * T * dt) * torch.randn(size=(N,))
        elif i == int(2*steps):
            # x_at_switch2 = x
            # x = x + dt * (-x + torch.matmul(J, phi(x))) #+ math.sqrt(2 * T * dt) * torch.randn(size=(N,))
            x = x - dt * torch.mv(-torch.eye(N) + (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
                2 * T * dt) * torch.randn(size=(N,))
        elif i > int(2 * steps):
            x = x + dt * (-x + torch.matmul(J, phi(x))) #+ math.sqrt(2 * T * dt) * torch.randn(size=(N,))+ input(0.,0.,0.2)
            # x = x - dt * torch.mv(-torch.eye(N) + (J * phiPrime(x)).t(), -x + torch.matmul(J, phi(x))) + math.sqrt(
            #     2 * T * dt) * torch.randn(size=(N,))
            x_at_switch2 = x
    x1 = [tensor[0].item() for tensor in x_list]
    x2 = [tensor[1].item() for tensor in x_list]
    x3 = [tensor[2].item() for tensor in x_list]
    x4 = [tensor[-3].item() for tensor in x_list]
    x5 = [tensor[-4].item() for tensor in x_list]
    x6 = [tensor[-5].item() for tensor in x_list]
    # x_at_switch2 = x_list[-1]

    # return x1, x2, x3, x_at_switch1, x_at_switch2, x_list
    return x1, x2, x3, x4, x5, x6, x_list

def lgvDynamics2(N, x, J, dt, gamma, s, T):  # gamma == 1, input s
    phi = np.tanh
    h = np.dot(J, phi(x))
    phiPrime = lambda x: 1 - np.tanh(x) ** 2
    x_prime = phiPrime(x)
    a = h - x
    S = np.ones(N) * s
    x = x + dt * (-x + h - gamma * x_prime * np.dot(J.T, a)) + S * dt + math.sqrt(
            2 * T * dt)*np.random.randn(N)
    return x