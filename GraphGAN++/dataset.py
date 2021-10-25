import os

import numpy as np
from scipy.io import loadmat
from utils import *
from torch.utils.data import Dataset


def load_data(data_k, flag):
    # 自己数据集下载
    data_array = torch.load('/home/ubnn/Ablation experiment/α-GCNGAN_YWJ/data_supernode/' + str(data_k) + '_' + 'train_superNode.pth')
    data_label = torch.load('/home/ubnn/Ablation experiment/α-GCNGAN_YWJ/data_supernode/' + str(data_k) + '_' + 'superNode_label_.pth')
    # 加载数据集的时候，将正负分开。
    positive = []
    positive_label = []

    negative = []
    negative_label = []
    for i, j in enumerate(data_label):
        if j == 0:
            positive.append(data_array[i].unsqueeze(0))
            positive_label.append(j.unsqueeze(0))
        else:
            negative.append(data_array[i].unsqueeze(0))
            negative_label.append(j.unsqueeze(0))
    # data_array = data_dict['shift_ASD_normal_train_BrainNet']  # data_array 的 shape 是 866*116*116
    if flag == 0:
        demo = torch.cat(positive, dim=0)
        demo_label = torch.cat(positive_label, dim=0)
    else:
        demo = torch.cat(negative, dim=0)
        demo_label = torch.cat(negative_label, dim=0)
    return demo, demo_label


class DataSet(Dataset):
    def __init__(self, adj, label):
        self.adj_all = adj
        self.labels = label

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        label = self.labels[idx]
        return adj, label

    def __len__(self):
        return len(self.labels)
