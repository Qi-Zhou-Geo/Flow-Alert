#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-03-31
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission


import numpy as np

def smooth_outliers(data_array, row_or_column="column", outliers=95, smooth_window=1, replace_method="median"):

    '''
    Remove outliers if isolated by neighbors.
    Notes: the [1, 1, 1, 1, 1, 1, 1] first and last of 1 will be removed

    Args:
        data_array: numpy array, 2D or 1D
        row_or_column: str, if time is row, feature is column, then please choose "column"
        outliers: int or float,
        smooth_window: int, the total window lenght is 2 * smooth_window + 1

    Returns:
        output: numpy array, same shape as input data_array,
        the outliers was replaced by q95
    '''

    output = data_array.copy()

    if row_or_column == "column":
        axis = 0 # column-wise
    elif row_or_column == "row":
        axis = 1 # row-wise
    else:
        raise ValueError(f"row_or_column={row_or_column} must be 'row' or 'column'.")


    q95 = np.percentile(data_array, axis=axis, q=outliers, keepdims=True)
    # 1 is True = outlier in each column > 95th percentile in this column
    greater_q95 = data_array > q95

    # 1 = Ture, 0 = False
    kernel = np.ones(2 * smooth_window + 1)


    if replace_method == "median":
        replaced_value = np.median(data_array, axis=axis, keepdims=True)
    elif replace_method == "Q5":
        replaced_value = np.percentile(data_array, axis=axis, q=5, keepdims=True)
    elif replace_method == "Q95":
        replaced_value = np.percentile(data_array, axis=axis, q=95, keepdims=True)
    else:
        raise ValueError(f"check the replace_method={replace_method}.")


    if axis == 0:  # column-wise
        for col_idx in range(data_array.shape[1]): # loop the column
            # convert bool to float for convolution
            col_data = greater_q95[:, col_idx].astype(float)
            conv_result = np.convolve(col_data, kernel, mode='same')

            # select the id when 1） data-60s > q95, and 2)
            isolated_mask = (col_data == 1) & (conv_result == 1)
            # replace with q95
            output[isolated_mask, col_idx] = replaced_value[0, col_idx]

            # handle boundaries (first and last `smooth_window` elements)
            for i in range(smooth_window):
                # left boundary side
                if col_data[i] == 1: # True
                    output[i, col_idx] = replaced_value[0, col_idx]

                # right boundary  side
                if col_data[-i] == 1:
                    output[-1 * i, col_idx] = replaced_value[0, col_idx]

    else:   # row-wise
        for row_idx in range(data_array.shape[0]):
            # convert bool to float for convolution
            row_data = greater_q95[row_idx, :].astype(float)
            conv_result = np.convolve(row_data, kernel, mode='same')

            # mask: Isolated outliers (True with 2 False neighbors on each side)
            isolated_mask = (col_data == 1) & (conv_result == 1)
            # replace with q95
            output[row_idx, isolated_mask]= replaced_value[row_idx, 0]

            # handle boundaries (first and last `smooth_window` elements)
            for i in range(smooth_window):
                # left boundary
                if row_data[i] == 1:
                    output[row_idx, i] = replaced_value[row_idx, 0]

                # right boundary
                if row_data[-i] == 1:
                    output[row_idx, -1 * i] = replaced_value[row_idx, 0]

    return output

