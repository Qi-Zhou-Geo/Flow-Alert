#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-08-16
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from datetime import datetime, timezone

from sklearn.metrics import confusion_matrix

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.visualize.heatmap_plot import probability_map

#p = "Luding-WD-2023-STA02-BHZ-E-testing-False"
#p = "Luding-LD-2023-STA01-DPZ-E-testing-False"
p = "Luding-AM-2024-R9BF5-EHZ-E-testing-False"

catchment_name, seismic_network, input_year, input_station, input_component, \
feature_type, dataloader_type, with_label = p.split("-")

fig = plt.figure(figsize=(5, 3.5))
gs = gridspec.GridSpec(2, 2, height_ratios=[30, 1])

# <editor-fold desc="probability">
ax = plt.subplot(gs[0])
cbar_ax = plt.subplot(gs[2])

df = pd.read_csv(f"{current_dir}/{p}-predicted.txt", header=0)
df_arr = np.array(df)
time_window_start = df_arr[:, 1]
pre_y_pro = df_arr[:, -2]

probability_map(ax=ax, time_window_start=time_window_start, visualization_value=pre_y_pro,
                vmin=0, vmax=1,
                fig=fig, cbar_ax=cbar_ax,
                cbar_ax_title="Debris flow probability")
ax.set_title(label=f"{p}", fontsize=7, weight='bold')
# </editor-fold>


# <editor-fold desc="rainfall">
ax = plt.subplot(gs[1])
cbar_ax = plt.subplot(gs[3])

selected_period = ['2023-06-06T00:00:00', '2023-11-03T23:00:00']
df = pd.read_csv(f"/Users/qizhou/#python/#GitHub_saved/Luding/data/precipitation_data/2023.txt")

df_arr = np.array(df)
date_arr = df_arr[:, 0]
id1 = np.where(date_arr == selected_period[0])[0][0]
id2 = np.where(date_arr == selected_period[1])[0][0] + 1

date_arr = df_arr[id1:id2, 0]
time_window_start = [datetime.strptime(str_time, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() for str_time in date_arr]

max_prcp = df_arr[id1:id2, 6]
mean_prcp = df_arr[id1:id2, 7]
min_prcp = df_arr[id1:id2, 8]

max_prcp[max_prcp == "no_nc_data"] = np.nan
max_prcp = max_prcp.astype(float)

mean_prcp[mean_prcp == "no_nc_data"] = np.nan
mean_prcp = mean_prcp.astype(float)

min_prcp[min_prcp == "no_nc_data"] = np.nan
min_prcp = min_prcp.astype(float)

probability_map(ax=ax, time_window_start=time_window_start, visualization_value=mean_prcp,
                vmin=0, vmax=10, cmap="Blues",
                fig=fig, cbar_ax=cbar_ax,
                cbar_ax_title="Precipitation [mm/h]")
# </editor-fold>

plt.tight_layout()
plt.savefig(f"{current_dir}/pro_{p}.png", dpi=600)
plt.show()

