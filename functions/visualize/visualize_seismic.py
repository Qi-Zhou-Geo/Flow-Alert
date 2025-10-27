#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-07-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import numpy as np
from datetime import datetime
from obspy import read, Trace, Stream, read_inventory, signal, UTCDateTime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def convert_st2tr(st):

    if isinstance(st, Stream):
        st = st[0]
    elif isinstance(st, Trace):
        pass
    else:
        print(f"!!! Error\n"
              f"Make sure the input for <convert_st2tr> is Obspy 'Trace' or 'Stream'.")

    return st

def rewrite_x_ticks(ax, data_start, data_end, data_sps, x_interval=1, align_to_minutes=15):

    start = UTCDateTime(data_start).timestamp
    end = UTCDateTime(data_end).timestamp

    # Align to specified minute interval (in seconds)
    align_seconds = align_to_minutes * 60

    # Round start up to next aligned time
    first_aligned_ts = np.ceil(start / align_seconds) * align_seconds

    # Generate tick locations from aligned time
    x_location = np.arange(first_aligned_ts, end + 1, 3600 * x_interval)

    x_ticks = []
    for j, k in enumerate(x_location):
        if j == 0:
            fmt = "%Y-%m-%d\n%H:%M:%S"
        else:
            fmt = "%H:%M"
        x_ticks.append(UTCDateTime(k).strftime(fmt))

    # Convert to sample positions relative to data_start
    x_location = (x_location - start) * data_sps

    ax.set_xticks(x_location, x_ticks)


def psd_plot(fig, ax, cbar_ax, st, fix_colorbar=True, per_lap=0.5, wlen=60, x_interval=1, max_plot_f=25):

    st = convert_st2tr(st)
    st.spectrogram(per_lap=per_lap, wlen=wlen, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    data_sps = 1 / (per_lap * wlen)

    if fix_colorbar is True:
        ax.images[0].set_clim(-180, -100) # from experiences

    ax.set_ylim(1, max_plot_f)
    y_location = np.arange(10, max_plot_f + 1, 10)
    y_location = np.concatenate((np.array([1]), y_location, np.array([max_plot_f])))
    y_location = np.unique(y_location)

    ax.set_yticks(y_location, y_location)
    ax.set_ylabel('Frequency [Hz]', weight='bold')

    cbar = fig.colorbar(ax.images[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Power Spectral Density (dB)")

    rewrite_x_ticks(ax,
                    data_start=st.stats.starttime,
                    data_end=st.stats.endtime,
                    data_sps=1, # fixed for psd
                    x_interval=x_interval)


    return ax, data_sps

def waveform_plot(ax, st, x_interval=1):

    st = convert_st2tr(st)
    data_source = f"{st.stats.network}-{st.stats.station}-{st.stats.channel}-SPS={int(st.stats.sampling_rate)}"

    ax.plot(st.data, color="black", label=data_source)
    ax.set_xlim(0, st.data.size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(st.stats.sampling_rate * 3600 * x_interval))  # unit is saecond
    ax.legend(loc="upper left", fontsize=6)
    ax.set_ylabel('Ampitude\n[m/s]', weight='bold')

    rewrite_x_ticks(ax,
                    data_start=st.stats.starttime,
                    data_end=st.stats.endtime,
                    data_sps=st.stats.sampling_rate,
                    x_interval=x_interval)

    return ax


def pro_plot(ax, pre_y_pro, ci_range, data_start, data_end, data_sps, plot_CI=True, x_interval=1):

    x = np.arange(pre_y_pro.size)
    ax.plot(x, pre_y_pro, color="black", label="Mean Pro", lw=1, zorder=2)

    if plot_CI is True:
        ax.fill_between(x, pre_y_pro - ci_range, pre_y_pro + ci_range, color="black", label="95% CI", alpha=0.5, zorder=2)

    ax.set_xlim(0, len(x))
    ax.set_ylim(0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1], [0.0, 0.25, 0.50, 0.75, 1.0])
    ax.legend(loc="upper left", fontsize=6)
    ax.set_ylabel('Probability', weight='bold')

    rewrite_x_ticks(ax,
                    data_start=data_start,
                    data_end=data_end,
                    data_sps=data_sps,
                    x_interval=x_interval)

    return ax
