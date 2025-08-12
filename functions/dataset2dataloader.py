#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir


def load_all_features(input_year, input_station, input_component, training_or_testing):

    # BL sets
    df1 = pd.read_csv(f"{project_root}/data_output/seismic_feature/{input_year}_{input_station}_{input_component}_all_A.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17])

    # waveform, spectrum, spectrogram sets
    df2 = pd.read_csv(f"{project_root}/data_output/seismic_feature/{input_year}_{input_station}_{input_component}_all_B.txt",
                      header=0, low_memory=False, usecols=np.arange(4, 63))

    # network sets
    df3 = pd.read_csv(f"{project_root}/data_output/seismic_feature/{input_year}_{input_component}_all_network.txt",
                      header=0, low_memory=False, usecols=np.arange(3, 13))

    if training_or_testing == "training" or training_or_testing == "testing":
        # time stamps, binary labels, probability of label 1(DF)
        df4 = pd.read_csv(f"{project_root}/data_output/event_label/"
                          f"{input_year}_{input_station}_{input_component}_observed_label.txt",
                          header=0, low_memory=False)
    elif training_or_testing == "dual_testing":
        # give a fake observed label with same length as df1-df3
        # time stamps, binary labels (fake), probability of label 1(DF)
        # you can use df1
        df4 = pd.read_csv(f"{project_root}/data_output/event_label/"
                          f"{input_year}_{input_station}_{input_component}_observed_label.txt",
                          header=0, low_memory=False)
    else:
        print(f"check the training_or_testing {training_or_testing}")


    df = pd.concat([df1, df2, df3, df4], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df3.columns.values, df4.columns.values])
    df.columns = columnsName

    return df


def select_features(input_station, feature_type, input_component, training_or_testing, input_data_year):
    '''
    Parameters:
    ----------
            input_station: str, input station
            feature_type: str, feature type (A, B, and C)
            input_component: str, data component
            input_data_year: list[float], data year

    Returns:
            input_features_name: data frame, feature name
            x: data frame, input data with corresponding timestamp
            y: data frame, data label with corresponding timestamp
            time_stamp_float: data frame in float type, corresponding timestamp
            time_stamp_str: data frame in float type, corresponding timestamp
    -------

    '''

    df = pd.DataFrame() # emoty df to store the data

    mapping_input_station = {"ILL18":"ILL08", "ILL12":"ILL02", "ILL13":"ILL03"}
    for input_year in input_data_year:
        if input_year == 2017:
            station = mapping_input_station.get(input_station)
        else:
            station = input_station
        df1 = load_all_features(input_year, station, input_component, training_or_testing)

        df = pd.concat([df, df1], axis=0, ignore_index=True)

    if feature_type == "A": # BL sets
        selected_column = np.arange(0, 11).tolist()
        selected_column.extend([80, 81, 82])
    elif feature_type == "B": # waveform, spectrum, spectrogram, and network sets
        selected_column = np.arange(11, 83).tolist()
    elif feature_type == "C": # A and B
        selected_column = np.arange(0, 83).tolist()
    elif feature_type == "D": # selected from C
        selected_column = [0, 8, 10, 23, 24, 35, 80, 81, 82]
    elif feature_type == "E":  # selected from C

        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        config_path = f"{project_root}/config/config_inference.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            selected_column = config["feature_type_E"]

        selected_column.extend([80, 81, 82])
        with_network = False
    elif feature_type.split("-")[0] == "X":  # selected based on feature IMP
        from config.config_dir import CONFIG_dir
        # "X-16-model" means that the Top 16 features are selected
        num_selected = int(feature_type.split("-")[1])
        feature_weight = np.load(f"{project_root}/trained_model/feature_imp/"
                                 f"{str(input_station).replace('1', '0')}-{feature_type.split('-')[2]}-C.npy")

        if descending is True:
            # descending order, then select top "num_selected"
            feature_weight_selected = np.sort(feature_weight)[::-1][:num_selected]
        else:
            # ascending order, then select top "num_selected"
            feature_weight_selected = np.sort(feature_weight)[:num_selected]

        # find the original index in the 1-80 feature index
        selected_column = []
        for i in feature_weight_selected:
            selected_column.append(np.where(feature_weight == i)[0][0])
        selected_column.sort()
        selected_column.extend([80, 81, 82])
        with_network = True
    elif feature_type.split("-")[0] == "R":  # randomly selected
        # "K-16-model" means that randomly selected 16 features
        num_selected = int(feature_type.split("-")[1])
        selected_column = generate_random_selected_feature_id(repeate, num_selected, num_total_feature=80)
        print(f"{seismic_network, input_year, input_station, input_component, feature_type}\n"
              f"{selected_column}")

        selected_column.extend([80, 81, 82])
        with_network = True
    else:
        print(f"please check the {feature_type}")

    df = df.iloc[:, selected_column]
    input_features_name = df.columns[:-3]
    
    
    x, y, time_stamp_float, time_stamp_str = df.iloc[:, :-3], df.iloc[:, -1], df.iloc[:, -2], df.iloc[:, -3]

    return input_features_name, x, y, time_stamp_float, time_stamp_str

# Assume C features are input
def select_specific_features(x, feature_names, selected_idxs):
    """
    Select specific features from the input data based on the given indices.

    Parameters:
    x : array-like, shape (n_samples, n_features)
        The input data from which features need to be selected.
    
    feature_names : list of str
        The list of all feature names corresponding to the columns of x.
    
    selected_idxs : list of int
        The indices of the features to be selected.
    
    Returns:
    x_selected : array-like, shape (n_samples, len(selected_idxs))
        The data containing only the selected features.
    
    selected_feature_names : list of str
        The names of the selected features.
    
    Raises:
    IndexError: If any index in selected_idxs is out of bounds for the given feature_names or x.
    """
    
    # Check that the number of features matches the length of feature_names
    if x.shape[1] != len(feature_names):
        raise ValueError(f"The number of features in x {x.shape[1]} does not match the length of feature_names {len(feature_names)}.")
    
    # Validate indices
    for idx in selected_idxs:
        if idx < 0 or idx >= len(feature_names):
            raise IndexError(f"Index {idx} is out of bounds for the feature_names list with length {len(feature_names)}.")
    
    # Extract the selected features from x using the given indices
    try:
        x_selected = x.iloc[:, selected_idxs]
    except IndexError as e:
        raise IndexError(f"Error selecting indices {selected_idxs} from x with shape {x.shape}: {e}")
    
    # Extract the names of the selected features
    selected_feature_names = [feature_names[i] for i in selected_idxs]
    
    return x_selected, selected_feature_names


def input_data_normalize(X_train, X_test):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def data2seq(df, seq_length):

    df_array = df.values.astype('float64')  # Convert DataFrame to NumPy array
    sequences = []

    for i in range(len(df_array) - seq_length - 1):

        features = df_array[i : i + seq_length, :-2]
        label =    df_array[i + seq_length, -2]
        time_stamps = df_array[i + seq_length, -1]

        sequences.append((features, label, time_stamps))

    return sequences


class seq2dataset(Dataset):
    # sequence to dataset
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        features, label, time_stamps = self.sequences[index]
        return dict(
            features=torch.Tensor(features),
            label=torch.tensor(label).to(torch.long),  # other type will raise error in loss function
            timestamps=torch.tensor(time_stamps)
        )


class dataset2dataloader:
    # dataset to dataloader
    def __init__(self, data_sequences, batch_size, training_or_testing):
        self.data_sequences = data_sequences
        self.batch_size = batch_size
        self.training_or_testing = training_or_testing
        self.dataset = seq2dataset(self.data_sequences)

    def dataLoader(self):

        if self.training_or_testing == "training":
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif self.training_or_testing == "testing" or self.training_or_testing == "dual_testing":
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        else:
            raise ValueError("training_or_testing must be 'training' or 'testing'")

        return data_loader