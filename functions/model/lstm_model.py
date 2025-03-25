#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import yaml

from scipy.stats import t as student_t  # Student's t-distribution

import torch
import torch.nn as nn
# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0

from pathlib import Path
from tqdm import tqdm


class LSTM_Classifier(nn.Module):
    def __init__(self, feature_size, device,
                 hidden_size=512, num_layers=4,
                 dropout=0.1, bidirectional=False, output_dim=2):
        super().__init__()

        # suppose input
        # "t" is time stamps of t to t_i
        # shape "t" = ([batch_size, sequence_length])
        # "x" is features of t to t_i
        # shape "x" = ([batch_size, sequence_length, feature_size])

        # "lstm" receive "x", and return "y" ([batch_size, output_dim])

        # initial parameters
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.output_dim = output_dim

        # same as "D = 2 if bidirectional=True otherwise 1" in Pytorch
        if self.bidirectional is True:
            self.D = 2
        else:
            self.D = 1

        # lstm layer
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)

        # fully connected layer
        self.fully_connect = nn.Linear(self.hidden_size * self.D, self.output_dim)

    def forward(self, x, t):

        x = x.to(torch.float32)
        self.lstm.flatten_parameters()

        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size, dtype=x.dtype).to(self.device)
        c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size, dtype=x.dtype).to(self.device)

        # output.shape = ([batch_size, sequence_length, hidden_size])
        # h_n.shape = c_n.shape = ([num_layers * self.D, batch_size, hidden_size])
        output, (h_n, c_n) = self.lstm(x, (h0, c0)) # do not need (h_n, c_n)


        # extract features
        if self.bidirectional is True:
            forward_last = h_n[-2, :, :]   # last layer's forward hidden state
            backward_last = h_n[-1, :, :]  # last layer's backward hidden state
            # concatenate along the feature dimension
            output = torch.cat((forward_last, backward_last), dim=1)
        else:
            #output = h_n[-1, :, :]
            # or use "output = output[:, -1, :]" to only focous the last time step in "sequence_length" domain
            output = output[:, -1, :]

        output = self.fully_connect(output)

        return output
