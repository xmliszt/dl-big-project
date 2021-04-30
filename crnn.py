import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn


class Pcrnn(nn.Module):
    def __init__(self, ks=(3, 1)):
        super().__init__()
        # CNN
        self.conv2d_1 = nn.Conv2d(1, 16, ks)
        self.conv2d_2 = nn.Conv2d(16, 32, ks)
        self.conv2d_3 = nn.Conv2d(32, 64, ks)
        self.conv2d_4 = nn.Conv2d(64, 64, ks)
        self.conv2d_5 = nn.Conv2d(64, 64, ks)

        self.maxPool2d_1 = nn.MaxPool2d((2, 2))
        self.maxPool2d_2 = nn.MaxPool2d((2, 2))
        self.maxPool2d_3 = nn.MaxPool2d((2, 2))
        self.maxPool2d_4 = nn.MaxPool2d((4, 4))
        self.maxPool2d_5 = nn.MaxPool2d((4, 4))

        self.flatten_1 = nn.Flatten()

        # RNN
        self.pool_lstm = nn.MaxPool2d((4, 2))
        self.gru = nn.GRU(input_size=64, hidden_size=1, bidirectional=True)

        # Overall
        self.linear = nn.Linear(576, 10)  # 10 genres
        self.softmax = nn.Softmax(dim=1)

    def forward_cnn(self, x: torch.Tensor):
        x = F.relu(self.conv2d_1(x))
        x = self.maxPool2d_1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.maxPool2d_2(x)
        x = F.relu(self.conv2d_3(x))
        x = self.maxPool2d_3(x)
        x = F.relu(self.conv2d_4(x))
        x = self.maxPool2d_4(x)
        x = F.relu(self.conv2d_5(x))
        x = self.maxPool2d_5(x)
        x = self.flatten_1(x)
        return x

    def forward_rnn(self, x: torch.Tensor):
        x = self.pool_lstm(x)
        x = x.view(x.size(0), x.size(2), x.size(3))
        x = self.gru(x)
        x = self.flatten_1(x[0])
        return x

    def forward(self, x: torch.Tensor):
        cnn = self.forward_cnn(x)
        rnn = self.forward_rnn(x)
        output = torch.cat([cnn, rnn], dim=1)
        output = self.linear(output)
        output = self.softmax(output)

        return output
