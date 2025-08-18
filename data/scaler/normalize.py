#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-08-05
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import sys
import yaml

import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime, timezone, date, timedelta

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.public.load_data import select_features


plt.rcParams.update( {'font.size':7,
                      'font.family': "Arial",
                      'axes.formatter.limits': (-8, 6),
                      'axes.formatter.use_mathtext': True} )


catchment_name = "Illgraben"
feature_type = "C"
station = 2 # 2 -> IGB/ILL 02 or 12 station
param = [
#"9J-2013-IGB0*-HHZ-training-True",
#"9J-2014-IGB0*-HHZ-training-True",
"Illgraben-9S-2017-ILL0*-EHZ-C-training-True",
"Illgraben-9S-2018-ILL1*-EHZ-C-training-True",
"Illgraben-9S-2019-ILL1*-EHZ-C-training-True",
"Illgraben-9S-2020-ILL1*-EHZ-C-testing-True",
"Illgraben-9S-2022-ILL1*-EHZ-C-testing-True",
]


for i, p in enumerate(param):

    p = p.replace("*", f"{station}")
    catchment_name, seismic_network, input_year, input_station, input_component, \
    feature_type, dataloader_type, with_label = p.split("-")


    input_features_name, data_array = select_features(catchment_name,
                                                      seismic_network,
                                                      input_year,
                                                      input_station,
                                                      input_component,
                                                      feature_type,
                                                      with_label,
                                                      repeat=1,
                                                      normalize=False)
    df = pd.DataFrame(df)

    if i == 0:
        df_all = df
    else:
        # concatenating df as new rows at the bottom of df_all
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)

df_all.to_csv(f"{project_root}/data/seismic_feature/2017-2022-02station-{catchment_name}-{feature_type}.txt", sep=',',
              index=False, index_label=False)


def stastic_arr(id_s, id_e, rms, iqr):
    
    rms_max = np.max(rms[id_s:id_e])
    rms_mean = np.mean(rms[id_s:id_e])
    rms_min = np.min(rms[id_s:id_e])
    
    iqr_max = np.max(iqr[id_s:id_e])
    iqr_mean = np.mean(iqr[id_s:id_e])
    iqr_min = np.min(iqr[id_s:id_e])

    output = np.array([rms_max, rms_mean, rms_min, iqr_max, iqr_mean, iqr_min])

    return output

def get_weekend_days(year):

    weekends = []
    d = date(year, 1, 1)
    delta = timedelta(days=1)

    while d.year == year:
        if d.weekday() == 5 or d.weekday() == 6:  # 5 = Saturday, 6 = Sunday
            weekends.append(d.strftime("%Y-%m-%dT%H:%M:%S"))
        d += delta

    return weekends


df_all = pd.read_csv(f"{project_root}/data/seismic_feature/2017-2022-02station-{catchment_name}-{feature_type}.txt", header=0)
df_all_arr = np.array(df_all)
df_all_date = df_all_arr[:, 0]
df_all_date = [datetime.fromtimestamp(i, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") for i in df_all_date]
df_all_date = np.array(df_all_date)


rms = df_all_arr[:, 34 + 1]
iqr = df_all_arr[:, 35 + 1]

df_event = pd.read_csv(f"{project_root}/data/manually_labeled_DF/Flow_Bench_Catalog.txt", header=0, nrows=65)
df_event_arr = np.array(df_event)

temp_event = []
for i in np.arange(df_event_arr.shape[0]):
    s, e = df_event_arr[i, 12], df_event_arr[i, 13]

    try:
        id_s = np.where(df_all_date == s)[0][0]
        id_e = np.where(df_all_date == e)[0][0]

        #s = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        #e = datetime.strptime(e, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

        #id_s = np.where(np.abs(df_all_date - s) == np.min(np.abs(df_all_date - s)))[0][0]
        #id_e = np.where(np.abs(df_all_date - e) == np.min(np.abs(df_all_date - e)))[0][0]

        #t = datetime.fromtimestamp(df_all_date[id_s], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        output = stastic_arr(id_s, id_e, rms, iqr)
        temp_event.append(output)
    except:
        print("error", i, s, e)

temp_event = np.array(temp_event)


### select the "noise" period
temp_noise = []

jk = 0
for year in [2017, 2018, 2019, 2020, 2022]:
    weekends = get_weekend_days(year)

    for i, s in enumerate(weekends):
        e = s # give a "fake" e
        try:
            id_s = np.where(df_all_date == s)[0][0]
            id_e = id_s + 5

            output = stastic_arr(id_s, id_e, rms, iqr)
            temp_noise.append(output)

            jk += 1
            print(jk, year, s, e)
        except:
            pass

temp_noise = np.array(temp_noise)


### plot the results
upper_b = 5e-5
lower_b = 2e-8

title_l = ["Event", "Noise"]
fig = plt.figure(figsize=(5.5, 5))
gs = gridspec.GridSpec(2, 2)

for i, arr in enumerate([temp_event, temp_noise]):

    if i == 0:
        ax = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[2])
    else:
        ax = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[3])

    rms_max = arr[:, 0]
    rms_mean = arr[:, 1]
    rms_min = arr[:, 2]

    sns.kdeplot(rms_max, ax=ax, fill=True, label="Max", clip=(0, None))
    sns.kdeplot(rms_mean, ax=ax, fill=True, label="Mean", clip=(0, None))
    sns.kdeplot(rms_min, ax=ax, fill=True, label="Min", clip=(0, None))

    print(title_l[i])
    print("max", np.max(rms_max), np.max(rms_mean), np.max(rms_min))
    print("mean", np.mean(rms_max), np.mean(rms_mean), np.mean(rms_min))
    print("min", np.min(rms_max), np.min(rms_mean), np.min(rms_min))

    ax.set_title(title_l[i], fontsize=7, fontweight='bold')
    ax.set_yscale('log')

    ax.legend(loc="upper right", fontsize=6)
    ax.set_ylabel("Value", fontweight='bold')
    ax.set_xlabel("RMS [m/s]", fontweight='bold')

    ax2.plot(rms_max, label="Max")
    ax2.plot(rms_mean, label="Mean")
    ax2.plot(rms_min, label="Min")
    ax2.set_yscale('log')
    ax2.set_ylim(1e-10, 1e-4)
    ax2.axhline(y=upper_b, color="black", ls="--")
    ax2.axhline(y=lower_b, color="black", ls="--")

    ax2.legend(loc="upper right", fontsize=6)
    ax2.set_ylabel("RMS [m/s]", fontweight='bold')
    ax2.set_xlabel("Event Index", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{current_dir}/RMS-KDE.png", dpi=600, transparent=True)
plt.show()



### find the normalization values
config_path = f"{project_root}/config/config_inference.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

selected = config[f'feature_type_{feature_type}']


df_all = pd.read_csv(f"{project_root}/data/seismic_feature/2017-2022-02station-{catchment_name}-{feature_type}.txt", header=0)
date = np.array([df_all.iloc[:, 0]])

id1 = np.where(date == datetime.strptime("2017-05-18T00:00:00", "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())[1][0]
id2 = np.where(date == datetime.strptime("2020-05-29T00:00:00", "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())[1][0]

features = np.array(df_all.iloc[id1:id2, 1:-1])
date_s = np.array(df_all.iloc[id1:id2, 0])

rms = features[:, 34]
features_s = features[:, selected]

upper = np.where(rms >= upper_b)[0]
lower = np.where(rms < lower_b)[0]

min_arr1 = np.min(features[upper], axis=0)#[selected]
mean_arr1 = np.mean(features[upper], axis=0)#[selected]
max_arr1 = np.max(features[upper], axis=0)#[selected]

min_arr2 = np.min(features[lower], axis=0)#[selected]
mean_arr2 = np.mean(features[lower], axis=0)#[selected]
max_arr2 = np.max(features[lower], axis=0)#[selected]

# 2013-2014, 2017-2020, 2020 ILL/IGB
note = f"This normalize factors are basde on 2017-2019 02/12 station, " \
       f"the max factor was defined as the mean of upper boundary (RMS >= {upper}), " \
       f"the min factor was defined as the mean of lower boudary (RMS < {lower})," \
       f"the used feature_type is {feature_type}."
np.savez(f"{project_root}/data/scaler/normalize_factor4{feature_type}.npz",
         min_factor=mean_arr2.reshape(1, len(selected)),
         max_factor=mean_arr1.reshape(1, len(selected)),
         note=note)


test = np.load(f"{project_root}/data/scaler/normalize_factor4{feature_type}.npz")
