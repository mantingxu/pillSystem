import torch
import torch.nn as nn
import numpy as np


class MyDBL(nn.Module):
    def __init__(self):
        super(MyDBL, self).__init__()
        self.linearOne = nn.Linear(36, 24)
        self.linearTwo = nn.Linear(36, 24)

    def forward(self, fx, fi):
        zx = self.linearOne(fx)
        zi = self.linearTwo(fi)
        hadamard = zx * zi
        res = nn.Tanh(hadamard)
        return res


class MyMLP(nn.Module):
    def __init__(self, inputDim):
        super(MyMLP, self).__init__()
        self.linearOne = nn.Linear(inputDim, 32)
        self.linearTwo = nn.Linear(32, 16)
        self.linearThree = nn.Linear(16, 3)

    def forward(self, f):
        o = self.linearOne(f)
        o = self.linearTwo(o)
        o = self.linearThree(o)
        return o


class DBLANet(nn.Module):
    def __init__(self, inputDim):
        super(DBLANet, self).__init__()
        self.dblOne = MyDBL()
        self.dblTwo = MyDBL()
        self.dblThree = MyDBL()
        self.mlp = MyMLP(inputDim)

    def forward(self, fx, f1, f2, f3):
        f1_prime = self.dblOne(fx, f1)
        f2_prime = self.dblTwo(fx, f2)
        f3_prime = self.dblThree(fx, f3)
        final_f = torch.hstack((f1_prime, f2_prime))
        final_f = torch.hstack((final_f, f3_prime))
        res = self.mlp(final_f)
        return res
