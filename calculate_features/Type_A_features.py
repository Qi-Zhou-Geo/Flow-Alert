#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-12-27
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

# the BL featuers are based on the following papers
# paper 1
# Sambridge Malcolm, Hrvoje Tkalčić, and A. Jackson.
# "Benford's law in the natural sciences."
# Geophysical research letters 37, no. 22 (2010).
# paper 2
# Qi Zhou, Hui Tang, Jens M Turowski, Jean Braun, Michael Dietze, Fabian Walter, Ci‐Jian Yang, and Sophie Lagarde.
# “Benford’s law as debris flow detector in seismic signals.”
# Journal of Geophysical Research: Earth Surface, 129, e2024JF007691.
# paper 3
# Newma Mark EJ.
# "Power laws, Pareto distributions and Zipf's law."
# Contemporary physics 46, no. 5 (2005): 323-351.

import math

import numpy as np
import pandas as pd
from scipy.stats import iqr, ks_2samp, chi2_contingency, mannwhitneyu


def get_num_magnitude(variable):
    '''
    Get the mangnitude n for x * 10^n

    Args:
        variable: float or int, number like 1234.567, -0.0123

    Returns:
        data-60s magnitude based on 10
    '''

    if variable == 0:
        magnitude = 0
    else:
        # positive, if the number > 10
        # negative, how many 0 in front of the first digit
        magnitude = math.floor(math.log10(abs(variable)))

    return magnitude

def calBL_feature(data, ruler, epsilon=1e-8):
    '''
    Calculate the Benford's Law related seismic features

    Args:
        data: 1D numpy array, the physical unit of the data-60s is um/s
        ruler: int or float,
        epsilon: float, avoid the zero division

    Returns:
        output: 1D numpy array,
    '''

    # BL theoretical value
    BL_frequency = [round(np.log10(1+1/i), 3) for i in range(1, 10)]

    # convert to int and ingore the data-60s less than 0
    data_abs = np.abs(data).astype(int)


    # if you see the warning,
    # it means either the data-60s is too small or ruler is too big
    # data-60s is too small = source-receiver too big? or event energy is too weak?
    data_selected = data_abs + ruler

    # <editor-fold desc="iq, max, min, magnitude range">
    iqr_value = iqr(data_selected)
    magnitude_range = get_num_magnitude(variable=iqr_value)
    iqr_value = float("{:.2f}".format(iqr_value))

    max_amp = np.max(data_selected) - ruler
    max_amp = float("{:.2f}".format(max_amp))
    min_amp = np.min(data_selected) - ruler
    min_amp = float("{:.2f}".format(min_amp))
    # </editor-fold>

    # <editor-fold desc="digit frequency">
    amp_data = pd.DataFrame(data_selected)
    amp_data = amp_data.astype(str)

    d = (amp_data.iloc[:, 0]).str[0: 1]
    d = list(d)

    digit_count = np.empty((0, 9))
    for digit in range(1, 10):
        first_digit = d.count(str(digit))
        digit_count = np.append(digit_count, first_digit)

    digit_frequency = digit_count / np.sum(digit_count)
    digit_frequency = [float('{:.3f}'.format(i)) for i in digit_frequency]
    # </editor-fold>

    # <editor-fold desc="goodness, ks, MannWhitneU, alpha">
    frequency = np.empty((0, 9))
    for a in range(0, 9):
        first_digit_frequency = pow((digit_frequency[a] - BL_frequency[a]), 2) / BL_frequency[a]
        frequency = np.append(frequency, first_digit_frequency)
    goodness = (1 - pow(sum(frequency), 0.5)) * 100
    goodness = float("{:.3f}".format(goodness))

    statistic, pvalue = ks_2samp(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    ks = float("{:.3f}".format(pvalue))  # pvalue

    statistic, pvalue = mannwhitneyu(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    MannWhitneU = float("{:.3f}".format(pvalue))  # pvalue

    if ks >= 0.95 and MannWhitneU >= 0.95:
        follow = 1  # follow BL
    else:
        follow = 0  # do not follow BL

    sum_d = []
    y_min = np.nanmin(data_selected)
    if y_min == 0:  # in csse of "divide by zero encountered in scalar divide"
        print("Warning!\n Data contain zero magnitude data.")
        y_min = 1
    else:
        pass

    for s in range(0, len(data_selected)):
        i = np.log(data_selected[s] / y_min)
        sum_d.append(i)

    alpha = 1 + len(data_selected) / (np.sum(sum_d) + epsilon)
    alpha = float("{:.4f}".format(alpha))
    # </editor-fold>

    output = np.array([max_amp, goodness, iqr_value, magnitude_range, alpha, ks, MannWhitneU, follow], dtype=float)
    output = np.concatenate((digit_frequency, output), axis=0) # stack as column

    return output
