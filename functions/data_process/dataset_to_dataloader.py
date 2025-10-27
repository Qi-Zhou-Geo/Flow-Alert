#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


# data_process data-60s to dataloader
def data_to_seq(array, seq_length, classification_or_prediction="classification"):
    '''

    Args:
        array: numpy array, shape by [time stamps, time(column 0) + waveform + DF pro (column -1)]
        seq_length: int, sequency length
        classification_or_prediction: str, control the "t_target" and "target"

    Returns:
        list
    '''

    if classification_or_prediction == "classification":
        picker = 1
    elif classification_or_prediction == "prediction":
        picker = 0

    arr = np.array(array, dtype='float64')
    sequences = []

    for i in range(len(arr) - seq_length - 1):

        t_features = arr[i : i + seq_length, 0] # float time stamps of features, t to t_{seq_length}
        features = arr[i : i + seq_length, 1:-1] # denoised waveform, t to t_{seq_length}

        t_target = arr[i + seq_length - picker, 0] # float time stamps of target, t_{seq_length+1 - picker}
        target = arr[i + seq_length - picker, -1] # debris flow probability, t_{seq_length+1 - picker}

        sequences.append((t_features, features, t_target, target))

    return sequences


class seq_to_dataset(Dataset):
    # sequence to dataset
    def __init__(self, sequences, data_type):
        self.sequences = sequences
        self.data_type = data_type

        if self.data_type == "feature":
            self.dtype = torch.long
        elif self.data_type == "waveform":
            self.dtype = torch.long #torch.float32 for regression
        else:
            print(f"check the training/testing data type in seq_to_dataset, self.data_type={self.data_type}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        t_features, features, t_target, target = self.sequences[index]

        block = dict(
            t_features=torch.tensor(t_features),
            features=torch.Tensor(features),

            t_target=torch.tensor(t_target),
            target=torch.tensor(target).to(self.dtype)
        )
        return block


class dataset_to_dataloader:
    def __init__(self,
                 dataset,
                 batch_size,
                 training_or_testing,
                 training_shuffle=True,
                 testing_shufflle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.training_or_testing = training_or_testing
        self.training_shuffle = training_shuffle
        self.testing_shufflle = testing_shufflle

    def dataLoader(self):

        if self.training_or_testing == "training":
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.training_shuffle, drop_last=True)
        elif self.training_or_testing == "testing" or "validation":
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.testing_shufflle, drop_last=True)
        else:
            raise ValueError("training_or_testing must be 'training' or 'testing'")

        return data_loader
