#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-26T13:19:05
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime, read, Stream, read_inventory

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# import the custom functions
# import the custom functions
from functions.visualize.visualize_seismic import waveform_plot, psd_plot, pro_plot


def process_data(f_min=1.0, f_max=25.0):
    
    # time now
    t_now = UTCDateTime.now()
    julday = t_now.julday - 1 # merge yesterday's data
    st_path = Path(project_root) / f"deploy/liveshow_cache/seismic/9S.ILL12.EHZ.{t_now.year}.{str(julday).zfill(3)}.mseed"
    
    st = read(st_path)
    tr = st.copy()
    
    tr.merge(method=1, fill_value='latest', interpolation_samples=0)
    tr._cleanup()
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.01)
    
    inv_path = Path(project_root) / f"deploy/9S_2026.xml"
    inv = read_inventory(inv_path)
    tr.remove_response(inventory=inv)      
       
    tr.filter("bandpass", freqmin=f_min, freqmax=f_max)
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.01)
    
    return tr


def plot_daily(tr=None, x_interval=4, 
               f_min=1.0, f_max=25.0, 
               pro_threshold=0.8,
               t1=None, t2=None):
    
    plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})

    if tr == None:
        tr = process_data(f_min, f_max)
    else:
        pass
    
    if t1 is not None and t2 is not None:
        tr.trim(UTCDateTime(t1), UTCDateTime(t2))
        
        
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(4, 1, height_ratios=[10, 1, 10, 10])

    ax = plt.subplot(gs[0])
    cbar_ax = plt.subplot(gs[1])
    psd_plot(fig, ax, cbar_ax, st=tr, fix_colorbar=True, per_lap=0.5, wlen=60, 
             x_interval=x_interval, max_plot_f=int(f_max))
    
    ax = plt.subplot(gs[2])
    ax = waveform_plot(ax, st=tr, x_interval=x_interval)
    
    ax = plt.subplot(gs[3])
    pro_path = Path(project_root) / f"deploy/liveshow_cache/pro/event_pro_{str(julday).zfill(3)}.txt"
    df = pd.read_csv(pro_path, header=0)
    if t1 is not None and t2 is not None:
        date_str = np.array(df.iloc[:, 1])
        id1 = np.where(date_str==t1)[0][0]
        id2 = np.where(date_str==t2)[0][0] + 1
        df = df.iloc[id1:id2, :]
    
    
    pre_y_pro = np.array(df["pro_mean"]).astype(float)
    ci_range = np.array(df["pro_ci"]).astype(float)
    data_start, data_end = df.iloc[0, 1], df.iloc[-1, 1]
    data_sps = 1 / (df.iloc[1, 0] - df.iloc[0, 0]) # type: ignore
    pro_plot(ax, pre_y_pro, ci_range, data_start, data_end, data_sps, plot_CI=True, x_interval=x_interval)
    
    ax.set_xlabel("UTC+0 Time", fontweight='bold')
    
    # add warning information
    first_warning_id = np.where(pre_y_pro >= pro_threshold)[0]
    if len(first_warning_id) > 0:
        # there may be warning
        first_warning_id = first_warning_id[0]
    
        first_warning = df.iloc[first_warning_id, 1]
        max_amp_time = tr[0].stats.starttime + np.argmax(tr[0].data) / tr[0].stats.sampling_rate
        max_amp_time = max_amp_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        label = (f"First Warning: UTC {first_warning}\n"
                 f"Probability Threshold: {pro_threshold}\n")
        ax.axvline(x=first_warning_id, color="green", ls="--", label=label, zorder=4)
        
        
        max_amp_time_id = (UTCDateTime(max_amp_time) - UTCDateTime(df.iloc[0, 1])) * data_sps # type: ignore
        print(max_amp_time_id)
        ax.axvline(x=max_amp_time_id, color="red", ls="-", label=f"Max Amplitude: {max_amp_time}", zorder=3)
        ax.legend(loc="upper left", fontsize=6)
        
    png_path = Path(project_root) / f"deploy/liveshow_cache/plots/event_pro_{str(julday).zfill(3)}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600)
    plt.show()
    plt.close(fig)



t1="2026-07-26T03:00:00"
t2="2026-07-26T09:00:00"
f_min=1.0
f_max=25.0
x_interval = 2

t_now = UTCDateTime.now()
julday = t_now.julday # merge yesterday's data

dir_sac = f'/Users/qizhou/#python/Flow-Alert/deploy/liveshow_cache/seismic/{julday}'
all_files = os.listdir(dir_sac)

    
st = Stream()
for i in all_files:
    st = st + read(f"{dir_sac}/{i}")

tr = st.copy()
tr.merge(method=1, fill_value='latest', interpolation_samples=0)
tr._cleanup()
tr.detrend("linear")
tr.detrend("demean")
tr.taper(max_percentage=0.01)

inv_path = Path(project_root) / f"deploy/9S_2026.xml"
inv = read_inventory(inv_path)
tr.remove_response(inventory=inv)      

tr.filter("bandpass", freqmin=f_min, freqmax=f_max)
tr.detrend("linear")
tr.detrend("demean")
tr.taper(max_percentage=0.01)

if t1 is not None and t2 is not None:
    tr.trim(UTCDateTime(t1), UTCDateTime(t2))

plot_daily(tr=tr, x_interval=2, f_min=1.0, f_max=25.0, pro_threshold=0.8, t1=t1, t2=t2)

