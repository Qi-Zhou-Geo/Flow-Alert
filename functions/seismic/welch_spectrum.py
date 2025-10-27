#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-04
#__author__ = Qi Zhou and Sibashish Dash, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission


import numpy as np
from scipy.signal import welch


def welch_psd(data, sampling_rate, f_min, f_max, segment_window=10, scaling="density", unit_dB=True):
    """
    Estimate power spectral density (PSD) using Welch’s method

    Args:
        data: 1D numpy array, Time series data (m/s).
        sampling_rate: int, sampling frequency (Hz).
        f_min, f_max: float, Min and max frequency (Hz).
        segment_window: float or int, window size for the data segment (seconds)
        scaling: str, welch methods, either "density" or "spectrum"
        unit_dB: bool,
                 If True, convert psd to "10 * np.log10(psd)"

    Returns:
        freq: 1D numpy array, Frequencies (Hz), filtered between f_min and f_max.
        psd: 1D numpy array, Power Spectral Density.

    """

    nperseg = sampling_rate * segment_window
    noverlap = nperseg // 2 # 50% overlap

    freq, psd = welch(data, fs=sampling_rate, scaling=scaling,
                      nperseg=nperseg, noverlap=noverlap, average='mean')

    if scaling == "density":
        # psd unit by (m/s)**2 / Hz
        psd_unit = "[(m/s)² / Hz]"
    elif scaling == "spectrum":
        # psd unit by (m/s)**2
        psd_unit = "[(m/s)²]"

    if unit_dB:
        psd = 10 * np.log10(psd)  # dB relative to 1 (m²/s²)/Hz

    # frequency filtering
    valid_idx = (freq >= f_min) & (freq <= f_max)
    freq = freq[valid_idx]
    psd = psd[valid_idx]

    return freq, psd, psd_unit

