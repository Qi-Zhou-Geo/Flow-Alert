#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-09-14T17:32:21
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission
import yaml
import numpy as np

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


def load_Illgraben_range():
    # this is based on training and testing
    # if you can not find this ".npz" file, pelase run "data/scaler/run_normalize.sh"
    with np.load(f"{project_root}/data/scaler/normalize_factor4C.npz", "r") as f:
        min_factor = f["min_factor"]  # shape as (1, 80)
        max_factor = f["max_factor"]  # shape as (1, 80)

    with open(f"{project_root}/config/config_inference.yaml", "r") as f:
        config = yaml.safe_load(f)
    feature_H_id = config["feature_type_H"]

    feature_H_names = [
        "alpha",
        "ES_0",
        "ES_1",
        "ES_2",
        "ES_3",
        "ES_4",
        "env_max_to_duration",
        "RMS",
        "IQR",
        "MaxFFT",
        "DistMaxMean",
        "DistMaxMedian",
    ]

    return min_factor, max_factor, feature_H_id, feature_H_names


def validate_factors(min_factor, max_factor, size=12):

    maximum = np.asarray(max_factor, dtype=float).reshape(-1).copy()
    minimum = np.asarray(min_factor, dtype=float).reshape(-1).copy()

    if maximum.size != size or minimum.size != size:
        raise ValueError(f"Expected {size} min and max factors.")

    if not np.isfinite(maximum).all() or not np.isfinite(minimum).all():
        raise ValueError("Normalization factors must be finite.")

    if np.any(maximum <= minimum):
        raise ValueError("Each max factor must exceed its min factor.")

    return maximum, minimum


def _make_factors(lower, upper, window_size):

    if not np.isfinite([lower, upper, window_size]).all():
        raise ValueError("Bounds and window_size must be finite.")

    if not (0 < lower < upper and window_size > 0):
        raise ValueError("Require 0 < lower < upper and window_size > 0.")

    # alpha
    alpha_min = 1.0
    alpha_max = 10.0

    # seismic energy: use the same bounds for all 5 frequency bands
    es_min = np.log10(lower * window_size)
    es_max = np.log10(upper * window_size)

    # env_max_to_duration
    env_min = lower / window_size
    env_max = upper / window_size

    # RMS and IQR: use the chosen clipping policy
    rms_irq_min = lower
    rms_irq_max = upper

    # MaxFFT, DistMaxMean and DistMaxMedian: leave unchanged
    MaxFFT_min = lower / 1e10
    MaxFFT_max = upper / 1e5
    dist_min = lower / 1e10
    dist_max = upper / 1e5

    # Order:
    # alpha,
    # ES_0, ES_1, ES_2, ES_3, ES_4,
    # env_max_to_duration,
    # RMS, IQR,
    # MaxFFT, DistMaxMean, DistMaxMedian

    min_factor = np.array(
        [alpha_min, *([es_min] * 5), env_min, rms_irq_min, rms_irq_min, MaxFFT_min, dist_min, dist_min],
        dtype=float,
    )
    max_factor = np.array(
        [alpha_max, *([es_max] * 5), env_max, rms_irq_max, rms_irq_max, MaxFFT_max, dist_max, dist_max],
        dtype=float,
    )

    max_factor, min_factor = validate_factors(min_factor=min_factor, max_factor=max_factor, size=12)

    return min_factor, max_factor


def dynamic_norm(
    df,
    Illgraben_rms_min=1e-7,
    Illgraben_rms_max=5e-4,
    theory_min=1e-8,
    theory_max=1e-3,
    window_size=60,
):

    # check the raw data
    if "RMS" not in df.columns:
        raise ValueError(f"RMS is not in df headers.\ndf.columns: {df.columns}")

    if df["RMS"].isna().any():
        raise ValueError("RMS contains NaN values.")

    # replace the infity
    rms = df["RMS"].to_numpy(dtype=float, copy=True)
    rms[np.isposinf(rms)] = theory_max
    rms[np.isneginf(rms)] = theory_min
    df["RMS"] = rms

    # find the min and max rms
    min_rms = np.min(rms)
    max_rms = np.max(rms)

    # check the normalize type
    # case 1, no doublt
    if min_rms > Illgraben_rms_max:
        # all above the Illgraben range
        # >> use
        upper = theory_max
        lower = Illgraben_rms_max

    # case 2, kind of cherry-picking,
    # but we assume the maximum is alway unknow
    elif Illgraben_rms_min <= min_rms <= Illgraben_rms_max < max_rms:
        # maximum part above the Illgraben max, minium part above the Illrgaben min
        # >> use
        upper = theory_max
        lower = min_rms

    # case 3, no doublt
    elif min_rms < Illgraben_rms_min and max_rms > Illgraben_rms_max:
        # maximum part above the Illgraben max, minium part below the Illrgaben min
        # >> use
        upper = theory_max
        lower = Illgraben_rms_min

    # case 4, kind of cherry-picking,
    # but we assume the minium is alway unknow
    elif min_rms < Illgraben_rms_min <= max_rms <= Illgraben_rms_max:
        # maximum part below the Illgraben max, minium part below the Illrgaben min
        # >> use
        upper = max_rms
        lower = theory_min

    # case 5,
    # no doublt
    elif max_rms < Illgraben_rms_min:
        # maximum part below the Illgraben min
        # >> use
        upper = Illgraben_rms_min
        lower = theory_min

    # case 6,
    # no doublt
    else:
        # all within the Illgraben range
        # >> use
        upper = Illgraben_rms_max
        lower = Illgraben_rms_min

        # all factor based on training
        min_factor, max_factor, feature_H_id, feature_H_names = load_Illgraben_range()
        return min_factor, max_factor

    # theory range, shape as (1, 12)
    new_min_factor, new_max_factor = _make_factors(lower=lower, upper=upper, window_size=window_size)
    for i in new_max_factor - new_min_factor:
        print("new", i)

    # all factor based on training
    min_factor, max_factor, feature_H_id, feature_H_names = load_Illgraben_range()
    min_factor[0][feature_H_id], max_factor[0][feature_H_id] = new_min_factor, new_max_factor

    # min_factor, max_factor = min_factor[0][feature_H_id], max_factor[0][feature_H_id]
    # for i in max_factor - min_factor:
    #     print("old", i)

    return min_factor, max_factor
