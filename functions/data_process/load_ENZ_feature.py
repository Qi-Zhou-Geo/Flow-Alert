#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-06-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

from datetime import datetime

# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
import numpy as np
import torch.optim as optim
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.data_process.load_data import select_features
from functions.data_process.dataset_to_dataloader import *
from functions.data_process.min_max_normalize_transformer import min_max_normalize
from functions.data_process.undersamp_training_data import *


def prepare_dataloader_ENZ(feature_type, batch_size, seq_length, noise2event_ratio, params, repeat=1):

    # empty list to store the dataloader
    train_sequences = []
    test_sequences = []
    validate_sequences = []

    # loop the input data-60s
    for p in params:

        seismic_network, input_year, input_station, input_component, dataloader_type, with_label = p.split("-")

        # convert str to bool
        if with_label == "True":
            with_label = True
        else:
            with_label = False

        # load data_array as [time_stamps, features, target]
        catchment_name = "Illgraben"

        temp = []
        for i, input_component in enumerate(["EHE", "EHN", "EHZ"]):
            input_features_name, data_array_temp = select_features(catchment_name,
                                                              seismic_network,
                                                              input_year,
                                                              input_station,
                                                              input_component,
                                                              feature_type,
                                                              with_label,
                                                              repeat=repeat,
                                                              normalize=True)
            temp.append(data_array_temp)

        data_array = np.hstack((temp[0][:, :-1],  # time stamps + feature E
                                temp[1][:, 1:-1], # feature N
                                temp[2][:, 1:]))  # feature Z + label

        # convert data-60s frame to data-60s loader
        sequences = data_to_seq(array=data_array, seq_length=seq_length)

        # add sequences to list
        if dataloader_type == "training":
            train_sequences = train_sequences + sequences
        elif dataloader_type == "testing":
            test_sequences= test_sequences + sequences
        elif dataloader_type == "validation":
            validate_sequences = validate_sequences + sequences
        else:
            print(f"check the dataloader_type={dataloader_type}")

    # sample the sequences for training
    if noise2event_ratio < 300:
        # sample the training data-60s, and make sure "Non-DF : DF = noise2event_ratio : 1"
        train_sequences = under_sample_seq(train_sequences, noise2event_ratio)
    else:
        # use all training data-60s
        pass

    # convert sequences to data-60s loader
    train_dataloader = []
    test_dataloader = []
    validate_dataloader = []

    # train data-60s laoder, len(train_sequences) == 1
    dataset = seq_to_dataset(sequences=train_sequences, data_type="feature")
    dataloader = dataset_to_dataloader(dataset=dataset,
                                       batch_size=batch_size,
                                       training_or_testing="training")
    train_dataloader.append(dataloader.dataLoader())

    # test data-60s laoder, len(test_sequences) == 1
    dataset = seq_to_dataset(sequences=test_sequences, data_type="feature")
    dataloader = dataset_to_dataloader(dataset=dataset,
                                       batch_size=batch_size,
                                       training_or_testing="testing")
    test_dataloader.append(dataloader.dataLoader())

    # check the data-60s loader length
    if len(train_dataloader) == 0:
        # give "fake" dataloader to make sure lstm_train_test works well
        train_dataloader.append(test_dataloader[0])
    else:
        pass

    if len(test_dataloader) == 0:
        # give "fake" dataloader to make sure lstm_train_test works well
        test_dataloader.append(train_dataloader[0])
    else:
        pass

    if len(validate_dataloader) == 0:
        # give "fake" dataloader to make sure lstm_train_test works well
        validate_dataloader.append(train_dataloader[0])
    else:
        pass

    return train_dataloader, test_dataloader, validate_dataloader

