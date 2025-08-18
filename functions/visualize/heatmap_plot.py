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

from datetime import datetime

from sklearn.metrics import confusion_matrix

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# <editor-fold desc="add the Arial font">
import platform
if platform.system() == 'Linux':
    from matplotlib import font_manager
    font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
# </editor-fold>

plt.rcParams.update( {'font.size':7,
                      'font.family': "Arial",
                      'axes.formatter.limits': (-4, 6),
                      'axes.formatter.use_mathtext': True} )

def probability_map(ax, time_window_start, visualization_value,
                    vmin=0, vmax=1, cmap='inferno',
                    fig=None, cbar_ax=None, cbar_ax_title=None):

    '''
    Plot a 2D heatmap to visualize the value.

    Args:
        ax: plt ax,
        time_window_start: numpy 1D array or list, float time stamps
        visualization_value: numpy 1D array or list, values at time_window_start
        vmin: float, min of the visualization_value
        vmax: float, max of the visualization_value
        fig: plt fig,
        cbar_ax: plt ax,
        cbar_ax_title: str, content of the color bar

    Returns:

    '''

    '''
    

    Args:
        ax: plt ax,
        time_window_start: numpy 1D array or list, float time stamps
        visualization_value: numpy 1D array or list, values at time_window_start

    Returns:

    '''


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
        'pro': visualization_value})

    df_probability = df.pivot(index="julian_day", columns="seconds_id", values="pro")
    df_probability.fillna(0, inplace=True)
    df_probability.sort_values(by='julian_day', ascending=False, axis=0, inplace=True)

    heatmap = sns.heatmap(df_probability, vmin=vmin, vmax=vmax,
                          square=False, cbar=False,
                          cmap=cmap, ax=ax)

    ax.set_ylabel(f'Julian Day [{datetime.utcfromtimestamp(time_window_start[1]).strftime("%Y")}]', weight='bold')
    ax.set_xlabel('Time [UTC+0]', weight='bold')

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(sps / 6))  # set the "set_minor_locator" as 6h
    ax.xaxis.set_major_locator(ticker.MultipleLocator(sps))  # set the "set_major_locator" as 1h
    ax.set_xticks([i * sps for i in np.arange(0, 5)],
                  [f"{str(i).zfill(2)}:00" for i in np.arange(0, 25, 6)],
                  ha="center", rotation=0)

    if cbar_ax is not None:
        cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
        cbar.set_label(f"{cbar_ax_title}", fontsize=6)

    return heatmap
