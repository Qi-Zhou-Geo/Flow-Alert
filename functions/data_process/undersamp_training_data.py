#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-12-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import warnings
import random
import numpy as np

def under_sample_array(array, noise2event_ratio):
    '''
    Under sample the array to match Non-DF : DF = noise2event_ratio : 1

    Args:
        array: 2D numpy array, either train_data or test_data,
              structure as [row in time, time_stamps + features + label]

        noise2event_ratio: int, equal to Non-DF/DF

    Returns:
        sampled array in 2D numpy array
    '''

    label_1_data = array[array[:, -1] == 1]  # DF
    label_0_data = array[array[:, -1] == 0]  # non-DF

    num_label_1 = len(label_1_data)
    num_label_0 = num_label_1 * noise2event_ratio

    if num_label_0 > len(label_0_data):
        warnings.warn("in function <under_sample_array>"
                      "the noise2event_ratio is too big, and not enough Non-DF data-60s,"
                      "return the original data-60s array", UserWarning)

        print(f"num_label_0={num_label_0}, len(label_0_data)={len(label_0_data)}, "
              f"num_label_1={num_label_1}, noise2event_ratio={noise2event_ratio}")

        temp = array

        return temp

    if num_label_0 < num_label_1:
        warnings.warn("in function <under_sample_array>,"
                      "the Non-DF label less than DF label", UserWarning)

        print(f"num_label_0={num_label_0}, < num_label_1={num_label_1}.")

    sampled_label_1 = label_1_data
    sampled_label_0 = label_0_data[np.random.choice(len(label_0_data), num_label_0, replace=False)]

    temp = np.vstack((sampled_label_1, sampled_label_0))
    # sort as time series order
    temp = temp[temp[:, 0].argsort()]

    return temp


def under_sample_seq(sequences, noise2event_ratio):
    '''
    Under sample the sequences to match Non-DF : DF = noise2event_ratio : 1

    Args:
        sequences: list of tuples, same as "sequences.append((t_features, features, t_target, target))" in "data_to_seq"

        noise2event_ratio: int, equal to Non-DF/DF

    Returns:
        sampled data-60s in list of tuples
    '''

    df_sequences = []
    non_df_sequences = []

    for seq in sequences:

        t_features, features, t_target, target = seq

        # if the target has any label DF label (1), treat it as DF
        if np.any(target == 1):
            df_sequences.append(seq)
        else:
            non_df_sequences.append(seq)

    num_label_1 = len(df_sequences)
    num_label_0 = num_label_1 * noise2event_ratio

    if num_label_0 > len(non_df_sequences):
        warnings.warn("in function <under_sample_seq>"
                      "the noise2event_ratio is too big, and not enough Non-DF data-60s,"
                      "return the original data-60s array", UserWarning)

        print(f"num_label_0={num_label_0}, len(label_0_data)={len(non_df_sequences)}, "
              f"num_label_1={num_label_1}, noise2event_ratio={noise2event_ratio}")

        temp = sequences

        return temp

    if num_label_0 < num_label_1:
        warnings.warn("in function <under_sample_seq>,"
                      "the Non-DF label less than DF label", UserWarning)

        print(f"num_label_0={num_label_0}, < num_label_1={num_label_1}.")

    sampled_label_1 = df_sequences
    sampled_label_0 = random.sample(non_df_sequences, num_label_0)

    temp = sampled_label_0 + sampled_label_1
    # sort as time series order
    temp = sorted(temp, key=lambda x: np.min(x[2]))  # x[2] is t_target (1D array)

    return temp
