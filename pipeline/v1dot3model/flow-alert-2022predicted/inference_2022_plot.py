#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-17
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns

from datetime import datetime

from obspy import read, Trace, Stream
from obspy.core import UTCDateTime

from sklearn.metrics import confusion_matrix, f1_score

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.warning_strategy.buffer_prediction import cal_buffered_cm

plt.rcParams.update( {'font.size':7,
                      'font.family': "Arial",
                      'axes.formatter.limits': (-4, 6),
                      'axes.formatter.use_mathtext': True} )

def print_increased_warning(time_window_start, pre_y_pro, DF_threshold=0.5):
    df1 = pd.read_csv(f"{current_dir}/data/9S-2022-DF.txt", header=0)
    df2 = pd.read_csv(f"{current_dir}/data/9S-2022-CD29-time.txt", header=0)

    for i in range(len(df2)):
        t1 = UTCDateTime(df1.iloc[i, 2][:-2]+"00").timestamp - 3600
        t2 = UTCDateTime(df2.iloc[i, 0])

        id1 = np.where(time_window_start == t1)[0][0]
        id2 = np.where(time_window_start == t2)[0][0]+1
        predicted = pre_y_pro[id1:id2]

        if (predicted >= DF_threshold).sum() > 1:
            # with warning
            id3 = np.where(pre_y_pro[id1:id2] >= DF_threshold)[0][0]
            id3 = id3 + id1
            t3 = time_window_start[id3]
            increased_warning = (t2.timestamp - t3)/60 #unit is minute

            print(f"increased_warning:{increased_warning} for {df1.iloc[i, 2]}, CD29={df2.iloc[i, 0]}, \n"
                  f"warning at: {UTCDateTime(t3)}")

def visualize_probability_map(ax, time_window_start, pre_y_pro):

    # x interval
    sps = time_window_start[1] - time_window_start[0] # unit by seconds
    sps = 3600 * 6 / sps # set the "set_minor_locator" as 6h

    # <editor-fold desc="time">
    time_window_start_str = np.array([datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") for ts in time_window_start])
    julian_day = np.array([datetime.utcfromtimestamp(ts).strftime("%j") for ts in time_window_start])
    julian_day = julian_day.astype(int)
    seconds_id = np.array([(
            datetime.utcfromtimestamp(ts).hour * 3600 +
            datetime.utcfromtimestamp(ts).minute * 60 +
            datetime.utcfromtimestamp(ts).second) for ts in time_window_start])
    # </editor-fold>

    df = pd.DataFrame({
        'time_window_start': time_window_start_str,
        'julian_day': julian_day,
        'seconds_id': seconds_id,
        'pro': pre_y_pro})

    df_probability = df.pivot(index="julian_day", columns="seconds_id", values="pro")
    df_probability.fillna(0, inplace=True)
    df_probability.sort_values(by='julian_day', ascending=False, axis=0, inplace=True)

    heatmap = sns.heatmap(df_probability, vmin=0, vmax=1, square=False, cbar=False,
                          cmap='inferno', ax=ax)

    ax.set_ylabel('Day of Year 2022', weight='bold')
    ax.set_xlabel('Time [UTC+0]', weight='bold')

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(sps / 6))  # set the "set_minor_locator" as 6h
    ax.xaxis.set_major_locator(ticker.MultipleLocator(sps))  # set the "set_major_locator" as 1h
    ax.set_xticks([i * sps for i in np.arange(0, 5)],
                  [f"{str(i).zfill(2)}:00" for i in np.arange(0, 25, 6)],
                  ha="center", rotation=0)

    return heatmap


def psd_plot(ax, ax_twin, st, data_start, data_end, time_window_start, pre_y_pro, ci_range, index=None):
    start = UTCDateTime(data_start).timestamp
    end = UTCDateTime(data_end).timestamp

    if end - start > 3600 * 6:
        x_interval = 2
    else:
        x_interval = 1

    st.trim(UTCDateTime(data_start), UTCDateTime(data_end))
    st.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    ax.images[0].set_clim(-180, -100)

    temp = f"{UTCDateTime(data_start).strftime('%Y-%m-%d')}, Day {UTCDateTime(data_start).julday}"
    #ax.text(x=0, y=20, s=f" {index}\n {temp}", weight="bold", color="black", fontsize=7, zorder=7,
            #bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle="round,pad=0.3"))
    ax.set_title(f"{index} {temp}", weight="bold", loc="left", fontsize=7)

    ax.set_ylim(1, 50)
    ax.set_yticks([1, 10, 20, 30, 40, 50], [1, 10, 20, 30, 40, 50])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(60 * 60 * 2))  # unit is saecond


    x_location = np.arange(start, end + 1, 3600 * x_interval)
    x_ticks = []
    for j, k in enumerate(x_location):
        if j == 0:
            fmt = "%H:%M"
        else:
            fmt = "%H:%M"
        x_ticks.append(datetime.utcfromtimestamp(int(k)).strftime(fmt))

    ax.set_xticks(x_location-start, x_ticks)


    id1 = np.where(np.abs(time_window_start - start) ==
                   np.min(np.abs(time_window_start - start)))[0][0]
    id2 = np.where(np.abs(time_window_start - end) ==
                   np.min(np.abs(time_window_start - end)))[0][0]

    pro = pre_y_pro[id1:id2]
    ci_range_95 = ci_range[id1:id2]
    ci_lower = pro-ci_range_95
    ci_lower[ci_lower<0] = 0

    ci_upper = pro+ci_range_95
    ci_upper[ci_upper > 1] = 1

    x = np.arange(pro.size) * 60

    ax_twin.plot(x, pro, color="white", lw=1, zorder=2)
    ax_twin.fill_between(x, ci_lower, ci_upper, color="white", alpha=0.5, zorder=2)

    ax_twin.set_ylim(0.05, 1.05)
    ax_twin.set_yticks([0, 0.25, 0.50, 0.75, 1], [0, 0.25, 0.50, 0.75, 1])



model_version = "v1dot3model"
feature_type = "H"
batch_size = 128
seq_length = 64
df = pd.read_csv(f"{project_root}/pipeline/{model_version}/test_2022_9repeat/"
                 f"Illgraben-9S-2022-ILL12-EHZ-{feature_type}-testing-True-{model_version}-H-b{batch_size}-s{seq_length}-9.txt", header=0)
arr = np.array(df)

time_window_start = arr[:, 0]
obs_y_pro = arr[:, 2].astype(float)
pre_y_pro = arr[:, -2].astype(float)
ci_range = arr[:, -1].astype(float)

obs_y_label = arr[:, 2].astype(int)
DF_threshold = 0.5
pre_y_label = (pre_y_pro >= DF_threshold).astype(int)

cm = confusion_matrix(obs_y_label, pre_y_label)
f1 = f1_score(obs_y_label, pre_y_label)
print(cm)
print(f1)

cm_buffered, f1_buffered = cal_buffered_cm(obs_y_label, pre_y_label)
print(np.sum(cm_buffered))

# do not consider the two events.
t1 = ["2022-06-02T00:00:00", "2022-06-02T04:00:00"]
t2 = ["2022-06-28T12:30:00", "2022-06-28T20:30:00"]
id1 = np.where(time_window_start == UTCDateTime(t1[0]))[0][0]
id2 = np.where(time_window_start == UTCDateTime(t1[1]))[0][0]
id3 = np.where(time_window_start == UTCDateTime(t2[0]))[0][0]
id4 = np.where(time_window_start == UTCDateTime(t2[1]))[0][0]

obs_y_label[id1:id2] = 0
obs_y_label[id3:id4] = 0
pre_y_label[id1:id2] = 0
pre_y_label[id3:id4] = 0

cm_buffered, f1_buffered = cal_buffered_cm(obs_y_label, pre_y_label, buffer_l=5, buffer_r=180)
print(np.sum(cm_buffered))


df = pd.read_csv(f"{project_root}/pipeline/{model_version}/test_2022_9repeat/"
                 f"Illgraben-9S-2022-ILL12-EHZ-{feature_type}-testing-True-{model_version}-H-b{batch_size}-s{seq_length}-9.txt", header=0)
arr = np.array(df)

time_window_start = arr[:, 0]
obs_y_pro = arr[:, 2].astype(float)
pre_y_pro = arr[:, -2].astype(float)
ci_range = arr[:, -1].astype(float)
print_increased_warning(time_window_start, pre_y_pro, DF_threshold=0.5)


fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.05])

ax = plt.subplot(gs[0:3, 0])
ax.set_title(f" (a)", loc="left", weight="bold", fontsize=7)

heatmap = visualize_probability_map(ax, time_window_start, pre_y_pro)
#ax.text(x=0, y=10, s=f" (a)", weight="bold", color="white", fontsize=7)

cbar_ax = plt.subplot(gs[6])
cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
cbar.set_label("Predicted Debris Flow Probability")


st_name = ["9S-ILL12-EHZ-2022-06-30T19:00:00.mseed",
           "9S-ILL12-EHZ-2022-06-28T12:30:00.mseed",
           "9S-ILL12-EHZ-2022-06-02T00:00:00.mseed"]

index = ["(b)", "(c)", "(d)"]
df = pd.read_csv(f"{current_dir}/data/9S-2022-DF-b-c.txt", header=0)

for id, (idx, idy)in enumerate(zip([1, 3, 5], index)):

    ax = plt.subplot(gs[idx])
    ax_twin = ax.twinx()

    st = read(f"{current_dir}/data/{st_name[id]}")

    data_start = df.iloc[id, 2]
    data_end = df.iloc[id, 3]
    psd_plot(ax, ax_twin, st, data_start, data_end, time_window_start, pre_y_pro, ci_range, index[id])

    if id+1 == len(st_name):
        ax.set_xlabel('Time [UTC+0]', weight='bold')

    if id == 1:
        ax.set_ylabel('Frequency [Hz]', weight='bold')
        ax_twin.set_ylabel("Debris Flow Probability", weight='bold')

    if id == 0:
        cd29 = UTCDateTime("2022-06-30T20:52:00").timestamp - UTCDateTime(data_start).timestamp
        ax.vlines(x=cd29, ymin=1, ymax=50, color="green", lw=1, zorder=5)



cbar_ax = plt.subplot(gs[7])
cbar = fig.colorbar(ax.images[0], cax=cbar_ax, orientation="horizontal")
cbar.set_label("Power Spectral Density (dB)")


plt.tight_layout()
plt.subplots_adjust(hspace=0.8, wspace=0.2)
plt.savefig(f"{current_dir}/inference-2022-{model_version}-{feature_type}.png", dpi=600)
plt.show()

