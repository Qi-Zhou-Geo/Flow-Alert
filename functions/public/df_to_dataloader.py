#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-08-05
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import yaml
import argparse

import numpy as np

from datetime import datetime

# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
import torch.optim as optim
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.public.soft_normalize import soft_scaler
from functions.public.load_data import select_features
from functions.public.dataset_to_dataloader import data_to_seq, seq_to_dataset, dataset_to_dataloader


def prepare_sequences(params, normalize, seq_length):

    # empty list to store the dataloader
    train_sequences = []
    test_sequences = []

    # loop the input data
    for p in params:

        catchment_name, seismic_network, input_year, input_station, input_component, \
        feature_type, dataloader_type, with_label = p.split("-")

        # convert str to bool
        if with_label == "True":
            with_label = True
        else:
            with_label = False

        # load data_array as [time_stamps, features, target]
        input_features_name, data_array = select_features(catchment_name,
                                                          seismic_network,
                                                          input_year,
                                                          input_station,
                                                          input_component,
                                                          feature_type,
                                                          with_label,
                                                          repeat=1,
                                                          normalize=normalize)

        # convert data frame to data loader
        sequences = data_to_seq(array=data_array, seq_length=seq_length)

        # add sequences to list
        if dataloader_type == "training":
            train_sequences = train_sequences + sequences
        elif dataloader_type == "testing":
            test_sequences = test_sequences + sequences
        else:
            print(f"check the dataloader_type={dataloader_type}")


    if len(train_sequences) != 0 :
        pass
    else:
        train_sequences = None
        print("!!! Notice, the train_sequences is empty")

    if len(test_sequences)  != 0 :
        pass
    else:
        test_sequences = None
        print("!!! Notice, the test_sequences is empty")


    return train_sequences, test_sequences



def prepare_dataloader(params, normalize, batch_size, seq_length, repeat=1):
    '''
    Load seismic features (df) and convert it to torch dataloader

    Args:
        params: str, format as "Illgraben-9S-2020-ILL1*-EHZ-E-testing-True"
        normalize： bool, normalize the data or not
        batch_size: float, 32, 64, 128
        seq_length: float, 32, 64, 128
        repeat: float, 1, 2, 3, this parameter is designed for reduce uncentrity

    Returns:
        train and test dataloader: List[DataLoader]
    '''

    # empty list to store the dataloader
    train_sequences = []
    test_sequences = []

    # loop the input data
    for p in params:

        catchment_name, seismic_network, input_year, input_station, input_component, \
        feature_type, dataloader_type, with_label = p.split("-")

        # convert str to bool
        if with_label == "True":
            with_label = True
        else:
            with_label = False

        # load data_array as [time_stamps, features, target]
        input_features_name, data_array = select_features(catchment_name,
                                                          seismic_network,
                                                          input_year,
                                                          input_station,
                                                          input_component,
                                                          feature_type,
                                                          with_label,
                                                          repeat=repeat,
                                                          normalize=normalize)

        # convert data frame to data loader
        sequences = data_to_seq(array=data_array, seq_length=seq_length)

        # add sequences to list
        if dataloader_type == "training":
            train_sequences = train_sequences + sequences
        elif dataloader_type == "testing":
            test_sequences = test_sequences + sequences
        else:
            print(f"check the dataloader_type={dataloader_type}")

    # convert sequences to data loader
    train_dataloader = []
    test_dataloader = []

    if len(train_sequences) != 0 :
        # train data-loader, len(train_sequences) == 1
        dataset = seq_to_dataset(sequences=train_sequences, data_type="feature")
        dataloader = dataset_to_dataloader(dataset=dataset,
                                           batch_size=batch_size,
                                           training_or_testing="training")
    else:
        dataloader = None
        print("!!! Notice, the train_dataloader is empty. \n")

    train_dataloader.append(dataloader.dataLoader())

    if len(test_sequences)  != 0 :
        # test data-loader, len(test_sequences) == 1
        dataset = seq_to_dataset(sequences=test_sequences, data_type="feature")
        dataloader = dataset_to_dataloader(dataset=dataset,
                                           batch_size=batch_size,
                                           training_or_testing="training")
    else:
        dataloader = None
        print("!!! Notice, the test_dataloader is empty")

    test_dataloader.append(dataloader.dataLoader())


    return train_dataloader, test_dataloader

