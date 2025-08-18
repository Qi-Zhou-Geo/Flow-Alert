#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-01-03
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import cochrans_q, mcnemar
from itertools import combinations

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.public.archive_data import dump_evaluate_matrix


def mcnemar_test(target, model1_predicted, model2_predicted, model1=None, model2=None, print_results=True):
    '''
    McNemar's test to determin which model is better.

    References:
        Rainio, Oona, Jarmo Teuho, and Riku Klén.
        "Evaluation metrics and statistical tests for machine learning."
        Scientific Reports 14, no. 1 (2024): 6086.

    Args:
        target: 1D numpy array or one column in dataframe, manually label or ground truth
        model1_predicted: 1D numpy array or one column in dataframe, model1 predicted label
        model2_predicted:  1D numpy array or one column in dataframe, model2 predicted label
        model1: bool or str, name of model 1
        model2: bool or str, name of model 2

    Returns:
        contingency_table: 2*2 dataframe that includes a, b, c, d
        b, c: float value,
        stat: float value, equal (b-c)*2/(b+c)
        mcnemar_p_value: float value,
    '''
    
    model1_corrected = (model1_predicted == target).astype(int)
    model2_corrected = (model2_predicted == target).astype(int)

    contingency_table = np.zeros((2, 2), dtype=int)
    for i in range(len(target)):
        if model1_corrected[i] == 1 and model2_corrected[i] == 1:
            contingency_table[0, 0] += 1  # a
        elif model1_corrected[i] == 1 and model2_corrected[i] == 0:
            contingency_table[0, 1] += 1  # b
        elif model1_corrected[i] == 0 and model2_corrected[i] == 1:
            contingency_table[1, 0] += 1  # c
        elif model1_corrected[i] == 0 and model2_corrected[i] == 0:
            contingency_table[1, 1] += 1  # d

    contingency_table = pd.DataFrame(contingency_table, columns=["model2_correct", "model2_in-correct"])
    contingency_table.index = ["model1_correct", "model1_in-correct"]

    b, c =  contingency_table.iloc[0, 1], contingency_table.iloc[1, 0]

    if print_results is True:
        if b > c:
            print(f"{contingency_table}, \n"
                  f"b({b}) > c({c}), model1({model1}) performed is <better> than model2({model2})")
        elif b < c:
            print(f"{contingency_table}, \n"
                  f"b({b}) > c({c}), model1({model1}) performed is <worse> than model2({model2})")
        else:
            print(f"b({b}) > c({c})")

    if b + c >= 20:  # for large sample sizes
        stat, mcnemar_p_value = mcnemar(x=contingency_table, exact=False, correction=True)
    else:  # for small sample sizes
        stat, mcnemar_p_value = mcnemar(x=contingency_table, exact=True, correction=True)

    return contingency_table, b, c, stat, mcnemar_p_value
