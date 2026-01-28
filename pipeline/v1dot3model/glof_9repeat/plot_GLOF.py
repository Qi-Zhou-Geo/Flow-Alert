#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-17
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission


import os
import yaml
import argparse

import pandas as pd
import numpy as np


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.visualize.heatmap_plot import visualize_probability_map
from functions.seismic.plot_obspy_st import rewrite_x_ticks


plt.rcParams.update({'font.size': 7,
                     'font.family': "Arial",
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

def plot_time_stamps(ax, data_start, first_surge, second_surge, sps):
    color_l = [f"C0", f"C1", f"C5"]

    for i in range(len(first_surge)):
        lag = UTCDateTime(first_surge[i]) - UTCDateTime(data_start)
        ax.axvline(lag * sps, color=color_l[i], ls="-", lw=1, zorder=1, label=f"First Surge Arrival")

        lag = UTCDateTime(second_surge[i]) - UTCDateTime(data_start)
        ax.axvline(lag * sps, color=color_l[i], ls="--", lw=1, zorder=1, label=f"Second Surge Arrival")

def plot_first_waring(sta, t_str, first_surge, second_surge, pro_mean, epsilon=0.5):

    warning_time_id = np.where(pro_mean >= epsilon)[0][0]
    t0 = t_str[warning_time_id]

    if sta == "NEP08":
        i = 0
    elif sta == "NEP07" or sta == "NEP06":
        i = 1
    elif sta == "NEP05" or sta == "NEP04":
        i = 2
    else:
        i = None
        print("Check the ref time")

    t1 = UTCDateTime(first_surge[i]) - UTCDateTime(t0)
    t2 = UTCDateTime(second_surge[i]) - UTCDateTime(t0)

    print(f"{sta}, increased warning at: {t0}, 1st surge: {t1/60 :.2f}, 2nd surge: {t2/60 :.2f}")


# from upstream to downstream
model_version = "v1dot3model"
seismic_sta_list = ["NEP08", "NEP07", "NEP06", "NEP10", "NEP05", "NEP04"]
subplot_index = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# Figure 1c Glacial lake outburst floods as drivers of fluvial erosion in the Himalaya
# the three time is the 1st and 2nd time to
# NEP08 (Hindi station), NEP07/06 (Chaku station), and NEP05/04 (Tyantali station),
first_surge = ["2016-07-05T15:16:41", "2016-07-05T15:23:29", "2016-07-05T15:28:20"]
second_surge = ["2016-07-05T15:31:04", "2016-07-05T15:40:33", "2016-07-05T15:48:31"]
data_start, data_end = "2016-07-05T15:00:00+00:00", "2016-07-05T19:00:00+00:00"



fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.1])
ax_pro = plt.subplot(gs[2, :])
ax_l = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]),
        plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])]

for idx, sta in enumerate(seismic_sta_list):

    if idx <= 2:
        row = 0
        column = idx
    else:
        row = 1
        column = idx - 3

    df = pd.read_csv(f"{current_dir}/output/"
                     f"Bothekoshi-XN-2016-{sta}-HHZ-H-testing-False-{model_version}-H-b128-s64-9.txt", header=0)

    t_str = df["t_str=None"].values
    time_window_start_float = df["t_target=None"].values
    pro_mean = df["pro_mean"].values
    ci_range = df["ci_range"].values

    ax = ax_l[idx]#plt.subplot(gs[row, column])
    heatmap = visualize_probability_map(ax, time_window_start_float, pro_mean, cbar=False, cbar_ax=None)

    if idx == 5:
        cbar_ax = plt.subplot(gs[:2, 3])
        cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation="vertical") # "horizontal"
        cbar.set_label(label=f"Predicted Probability", fontsize=6, weight='bold')

    ax.set_title(label=f"{subplot_index[idx]} {sta}", loc="left", fontsize=7, weight='bold')

    if idx == 0 or idx ==3:
        ax.set_ylabel(f"Day of Year 2016", weight='bold')
    else:
        ax.set_ylabel(f"", weight='bold')

    if idx in [3, 4, 5]:
        ax.set_xlabel(f"Time [UTC+0]", weight='bold')
    else:
        ax.set_xlabel(f"", weight='bold')

    if sta in ["NEP08", "NEP07", "NEP05"]:
        plot_first_waring(sta, t_str, first_surge, second_surge, pro_mean, epsilon=0.5)


    # pro plot
    id1 = np.where(t_str == data_start)[0][0]
    id2 = np.where(t_str == data_end)[0][0] + 1
    x = np.arange(0, id2 - id1)
    y = pro_mean[id1:id2]
    ax_pro.plot(x, y, color=f"C{idx}", zorder=3, label=f"{sta}")
    y1 = y + ci_range[id1:id2]
    y2 = y - ci_range[id1:id2]
    ax_pro.fill_between(x, y1, y2, color=f"C{idx}", alpha=0.5, zorder=2)

    ax_pro.set_xlim(0, x[-1])
    ax_pro.set_ylim(0, 1.05)

    id1 = np.where(t_str == data_start)[0][0]
    id2 = np.where(t_str == data_end)[0][0]
    false_detection = np.sum(pro_mean[:id1] >= 0.5) + np.sum(pro_mean[id2:] >= 0.5)
    print(f"sta: {sta}, false detection: {false_detection}")

handles, labels = ax_pro.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax_pro.legend(unique_labels.values(), unique_labels.keys(), loc="best", fontsize=6, ncol=1)
plot_time_stamps(ax_pro, data_start, first_surge, second_surge, sps=1/60)

ax_pro.set_title(label=f"(g)", loc="left", fontsize=7, weight='bold')
ax_pro.set_ylabel(f"Predicted Probability", weight='bold')
ax_pro.set_xlabel('Time [UTC+0]', weight='bold')
ax_pro.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
rewrite_x_ticks(ax=ax_pro,
                data_start=data_start,
                data_end=data_end,
                data_sps=1/60,
                x_interval=1)

plt.tight_layout()
plt.savefig(f"{current_dir}/GLOF_one_month_test_{model_version}.png", dpi=600, transparent=True)
plt.show()
plt.close(fig=fig)



