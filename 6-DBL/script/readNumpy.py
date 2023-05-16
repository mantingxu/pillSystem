import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

file = np.load('../csv/myResTest.npy', allow_pickle=True)


# example of dataset create
class ExampleDataset(Dataset):
    # data loading
    def __init__(self):
        data = np.load('../csv/myResTest.npy', allow_pickle=True)
        self.data = data

    # working for indexing
    def __getitem__(self, index):
        fx = self.data[index][0]
        f1 = self.data[index][1]
        f2 = self.data[index][2]
        f3 = self.data[index][3]
        label = self.data[index][4]
        return fx, f1, f2, f3, label

    # return the length of our dataset
    def __len__(self):
        return len(self.data)


dataset = ExampleDataset()
#
# # pick first data
first_data = dataset[0]
fx, f1, f2, f3, label = first_data

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)

count = 0
for data in train_loader:
    print('num', count)
    print('data',data[0].shape)
    count+=1
    # print('batchsize', batchsize)


