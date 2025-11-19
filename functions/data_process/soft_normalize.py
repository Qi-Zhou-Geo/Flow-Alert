#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-08-05
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions

def get_rank_index(lower_boundary, upper_boundary, mean_rms, current_rms):

    temp_arr = np.array([lower_boundary, upper_boundary, mean_rms, current_rms])
    # argsort twice gives rank
    ranks = np.argsort(temp_arr) # index of a ranked from min to max

    return ranks


def load_normalize_factor(feature_type):

    with open(f"{project_root}/config/config_inference.yaml", "r") as f:
        config = yaml.safe_load(f)
    selected = config[f'feature_type_{feature_type}']

    with np.load(f"{project_root}/data/scaler/normalize_factor4C.npz", "r") as f:
        min_factor = f["min_factor"][selected]
        max_factor = f["max_factor"][selected]

    return min_factor, max_factor


def normalize_within_boundary(current_features, feature_type="E"):

    min_factor, max_factor = load_normalize_factor(feature_type)
    # min-max normalize
    current_features = (current_features - min_factor) / (max_factor - min_factor)

    return current_features



def normalize_across_boundary(data_array):

    # min-max normalize
    data_array = (data_array - min_factor) / (max_factor - min_factor)

    return data_array



def normalize_outside_boundary(case, prior_features, rms_id, boundary):

    if case == 1:
        id = np.where(prior_features[:, rms_id] == np.min(prior_features[:, rms_id]))[0][0]
    elif case == 2:
        pass

    scaler = prior_features[id, :]
    if np.any(scaler == 0):
        print("!!! Warning, scaler contains zero")

    x_scaled = (prior_features / scaler) * boundary
    x_scaled = x_scaled[-1, :]

    return x_scaled


def soft_scaler(prior_features, rms_id=6, lower_boundary=2e-8, upper_boundary=5e-5):

    '''
       Soft-normalize/scale 'current_rms' using
       empirical boundaries ('lower_boundary' and 'upper_boundary') and prior values (prior_features).

       The lower and upper RMS (unit by m/s) boundaries are derived from Illgraben field observations:
         - 2013–2014: station IGB02
         - 2017:      station ILL02
         - 2018–2020: station ILL12
         - 2020:      station ILL12

       Based on these, we assume 'current_rms' can fall into 5 status levels.
       This function 'soft_scaler' the values in 'current_rms' using:
         - a moving baseline from `prior_features`
         - the `lower_boundary` and `upper_boundary` for scaling

       Args:
           current_rms (np.ndarray or list): The current set of data to normalize.
           prior_features (np.ndarray or list): Historical data used as baseline reference.
           lower_boundary (float): Lower empirical threshold.
           upper_boundary (float): Upper empirical threshold.

       Returns:

    '''

    min_factor, max_factor = load_normalize_factor(feature_type)


    prior_features = np.array(prior_features)
    
    mean_rms = np.min(prior_features[:, rms_id])
    current_rms = prior_features[-1, rms_id]
    current_features = prior_features[-1, :]

    ranks = get_rank_index(lower_boundary, upper_boundary, mean_rms, current_rms)

    if ranks == np.array([2, 3, 0, 1]) or ranks == np.array([3, 2, 0, 1]):
        # Case 1: below the boundary
        case = 1

        current_features = normalize_outside_boundary(case,
                                                      prior_features,
                                                      rms_id,
                                                      boundary=min_factor)

    elif ranks == np.array([2, 0, 3, 1]) or ranks == np.array([3, 0, 2, 1]):
        # Case 2: entry into the boundary from below
        case = 2
    elif ranks == np.array([0, 2, 3, 1]) or ranks == np.array([0, 3, 2, 1]):
        # Case 3: within the boundary
        case = 3
        current_features = normalize_within_boundary(current_features, feature_type="E")
    elif ranks == np.array([0, 2, 1, 3]) or ranks == np.array([0, 3, 1, 2]):
        # Case 4: leave the boundary from the above
        case = 4
    elif ranks == np.array([0, 1, 2, 3]) or ranks == np.array([0, 1, 3, 2]):
        # Case 5: above the boundary
        current_features = normalize_outside_boundary(x=current_features,
                                                      boundary=max_factor)
    elif ranks == np.array([2, 0, 1, 3]) or ranks == np.array([3, 0, 1, 2]):
        # Case 6: boundary within the new data
        case = 6
        print(f"!!! Warning, please check the input data, "
              f"default boundary [{lower_boundary}, {upper_boundary}] is within your data")
    else:
        case = 7
        print(f"!!! Warning, your data does not belong to any of the default cases,"
              f"please check the input data {current_rms, prior_features, lower_boundary, upper_boundary}")


    return current_features


def scale_with_min_override(x, m, n):

    x = np.array(x, dtype=float)

    # Step 1: override min(x) if needed
    x_min = x.min()
    x_max = x.max()

    effective_min = max(x_min, m)  # pretend min is at least m

    # Step 2: avoid divide-by-zero
    if x_max == effective_min:
        return np.full_like(x, (m + n) / 2)

    # Step 3: scale into [m, n]
    scaled = (x - effective_min) / (x_max - effective_min) * (n - m) + m
    return scaled