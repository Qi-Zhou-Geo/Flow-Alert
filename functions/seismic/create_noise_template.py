#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-06-19
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import pandas as pd
import numpy as np

year = 2022
station = "ILL12"
componment = "EHZ"
file_dir = f"/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature_25/European/Illgraben/{year}/{station}/{componment}"
julday_day1 = 135
julday_day2 = 275

arrays = []
for i in range(julday_day1, julday_day2 + 1):
    df = pd.read_csv(f"{file_dir}/{year}_{station}_{componment}_{i}_B.txt")
    arrays.append(df.iloc[:, 4:].values)

arrays = np.stack(arrays)  # (n_days, n_time, n_features)
# calculate the mean by time-aware alignment
mean_arr = arrays.mean(axis=0)  # (n_time, n_features)


df1 = df.iloc[:, :4]
df2 = pd.DataFrame(mean_arr)
# create new mean arr df
temp_df = pd.concat([df1, df2], axis=1)
temp_df.columns = df.columns
# remove the date and only keep the time, 00:00:00, 00:01:00
for i in range(temp_df.shape[0]):
    temp_df.iloc[i, 0] = temp_df.iloc[i, 0][11:]
    temp_df.iloc[i, 1] = i

# save it
temp_df.to_csv(f"{file_dir}/{year}_{station}_{componment}_all_B_mean_values_from_{julday_day1}-{julday_day2}.txt", index=False)


# restack the arr
empty_arr = []
for i in range(julday_day1, julday_day2+1):
    df = pd.read_csv(f"{file_dir}/{year}_{station}_{componment}_{i}_B.txt", header=0)
    df.iloc[:, 4:] = df.iloc[:, 4:] - temp_df.iloc[:, 4:]
    empty_arr.append(df)

year_df = pd.concat(empty_arr, axis=0, ignore_index=True)
year_df.to_csv(f"{file_dir}/{year}_{station}_{componment}_all_B_minus_mean.txt", index=False)
