#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd

from typing import List

from datetime import datetime
from obspy import Stream, Trace
from obspy.core import UTCDateTime # default is UTC+0 time zone

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.seismic.st2tr import stream_to_trace

plt.rcParams.update( {'font.size':7,
                      'axes.formatter.limits': (-3, 6),
                      'axes.formatter.use_mathtext': True} )

def rewrite_x_ticks(ax, data_start, data_end, data_sps, x_interval=1):
    '''
    Re write the x/time ticks

    Args:
        ax:
        data_start:
        data_end:
        data_sps:
        x_interval: for PSD, levea it as 1, for waveform or other, set as SPS

    Returns:

    '''
    start = UTCDateTime(data_start).timestamp
    end = UTCDateTime(data_end).timestamp

    x_location = np.arange(start, end + 1, 3600 * x_interval)
    x_ticks = []
    for j, k in enumerate(x_location):
        if j == 0:
            fmt = "%Y-%m-%d\n%H:%M:%S"
        else:
            fmt = "%H:%M"
        x_ticks.append(datetime.utcfromtimestamp(int(k)).strftime(fmt))

    x_location = (x_location - start) * data_sps

    ax.set_xticks(x_location, x_ticks)

def psd_plot(fig, ax, cbar_ax, st, fix_colorbar=True, per_lap=0.5, wlen=60, x_interval=1):

    st = stream_to_trace(st)
    st.spectrogram(per_lap=per_lap, wlen=wlen, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    data_sps = 1 / (per_lap * wlen)

    if fix_colorbar is True:
        ax.images[0].set_clim(-180, -100) # from experiences

    ax.set_ylim(1, 45)
    ax.set_yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
    ax.set_ylabel('Frequency [Hz]', weight='bold')

    cbar = fig.colorbar(ax.images[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Power Spectral Density (dB)")

    rewrite_x_ticks(ax,
                    data_start=st.stats.starttime,
                    data_end=st.stats.endtime,
                    data_sps=1, # fixed for psd
                    x_interval=x_interval)


    return ax, data_sps


def plot_st(fig, st, time=None):
    '''
    plot the st, either one stream, or multiple streams

    Args:
        output_dir: str,
        output_format: str
        fig: fig
        st: obspy st
        time: str time stamps in list

    Returns:

    '''

    if type(st) is Trace:
        st = Stream([st])


    st.plot(fig=fig, handle=True, equal_scale=False, zorder=2)

    ax_list = []
    for idx in range(len(st)):
        ax_list.append(fig.axes[idx])

    if time is not None:
        for ax in ax_list:
            ax.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

            for idx, t in enumerate(time):
                ax.axvline(pd.to_datetime(t), color='r', linestyle='-', lw=1, zorder=1, label=t)

    # for y label
    fig.text(x=0, y=0.5, s="Amplitude [m/s]", weight='bold', va='center', rotation='vertical')
    # for x label
    fig.text(x=0.5, y=0, s="Time [UTC+0]", fontweight="bold", ha='center')

    return fig, ax_list


def time_series_plot2(amp_data, data_start, data_end,
                     x_interval=2, sps1=100, sps2=1/30):

    fig = plt.figure(figsize=(5.5, 4))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(amp_data, color="black", label="umap_component2")
    ax.set_ylabel("Amplitude [m/s]", fontweight='bold')
    ax.set_xlim(0, amp_data.size)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * sps1))  # second * hour * sps

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(umap_component1, color="black", label="umap_component1")
    ax.set_ylabel("UMAP Dimension 1", fontweight='bold')
    ax.set_xlim(0, umap_component1.size)
    ax.set_ylim(-50, 100)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * sps2))  # second * hour * sps


    duration = int((UTCDateTime(data_end) - UTCDateTime(data_start)) / 3600)
    xLocation = np.arange(0, sps2 * 3600 * (duration + x_interval), sps2 * 3600 * x_interval)
    xTicks = [(UTCDateTime(data_start) + i * 1/sps2).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
    ax.set_xticks(xLocation, xTicks)
    ax.set_xlabel(f"Time [UTC+0]", fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"{CONFIG_dir['parent_dir']}/plotting/umap-plot/time-umap.png", dpi=600)
    plt.close(fig)


def time_series_plot(stream, time_markers=None, time_markers_label=None):

    '''
    Plot the time seris data-60s

    Args:
        stream: Obspy Stream, collects all positional arguments into tuple
        time_markers: list of string time stamps, format by "%Y-%m-%dT%H:%M:%S"
        time_markers_label: custom explanation of your time stamps

    Returns:
        fig, then you can save it to somewhere
    '''

    stream_length = len(stream)
    assert len(time_markers) == len(time_markers_label), f"check the length time_markers and time_markers_label"

    fig = plt.figure(figsize=(5.5, 2 * stream_length))
    gs = gridspec.GridSpec(stream_length, 1)
    axes = []

    for idx, st in enumerate(stream):

        if type(st) is Stream:
            st.merge(method=1, fill_value='latest', interpolation_samples=0)
            st._cleanup()
            st = st[0]

        data = st.data
        sps = st.stats.sampling_rate
        duration = st.stats.endtime - st.stats.starttime # unit by second

        if duration >= 12 * 3600:
            x_interval = 4 * 3600
        elif 4 * 3600 <= duration < 12 * 3600:
            x_interval = 2 * 3600
        elif 1 * 3600 < duration < 4 * 3600:
            x_interval = 1 * 3600
        elif 0.5 * 3600 < duration <= 1 * 3600:
            x_interval = 0.25 * 3600 # 15 min
        else:
            x_interval = 5/60 * 3600 # 5 min

        ax = plt.subplot(gs[idx])
        axes.append(ax)

        ax.plot(data, color="black")
        ax.set_xlim(0, data.size)

        if time_markers  is not None and time_markers_label is not None:
            color_list = [f"C{i}" for i in range(len(time_markers))]

            for c, (t, s) in enumerate(zip(time_markers, time_markers_label)):
                # give different ls
                if s[:2] == "on":
                    ls = "--"
                else:
                    ls = "-"

                x = (UTCDateTime(t) - st.stats.starttime) * sps
                ax.axvline(x=x, lw=1, ls=ls, color=color_list[c], zorder=1, label=s)

        ax.legend(fontsize=6)

        xLocation = np.arange(0, st.stats.npts, sps * x_interval)
        xTicks = [(st.stats.starttime + i * st.stats.delta).strftime('%H:%M:%S') for i in xLocation]

        ax.set_xticks(xLocation, xTicks)
        ax.set_xlabel(f"Time [UTC+0, {st.stats.starttime.strftime('%Y-%m-%d')}]", fontweight='bold')


    return fig, axes
