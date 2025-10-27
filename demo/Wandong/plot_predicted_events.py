#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-08-16
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import shutil
import argparse

import logging

import pandas as pd
import numpy as np

from datetime import datetime
from obspy import read, Trace, Stream, read_inventory, signal, UTCDateTime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# import the custom functions
from functions.seismic.seismic_data_processing import load_seismic_signal
from functions.visualize.visualize_seismic import convert_st2tr, psd_plot, waveform_plot, pro_plot


def plot_3_subplot(pro_mean, pro_95ci_range, st, threshold=0.5):

    plt.rcParams.update({'font.size': 7,
                         'font.family': "Arial",
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})


    st = convert_st2tr(st)

    fig = plt.figure(figsize=(5.5, 5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[5, 5, 5, 1])


    ax = plt.subplot(gs[0])
    select_start_time = st.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")
    select_end_time = st.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S")
    pro_plot(ax, pre_y_pro=pro_mean, ci_range=pro_95ci_range,
             data_start=select_start_time, data_end=select_end_time,
             plot_CI=True,
             data_sps=1 / 60)
    ax.axhline(y=threshold, color='red', linestyle='-', lw=1)

    ax = plt.subplot(gs[1])
    waveform_plot(ax, st, x_interval=1)

    ax = plt.subplot(gs[2])
    cbar_ax = plt.subplot(gs[3])
    psd_plot(fig, ax, cbar_ax, st, fix_colorbar=True, per_lap=0.5, wlen=60, x_interval=1, max_plot_f=50)

    ax.set_xlabel('Time [UTC+0]', weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    output_dir = f"{current_dir}/plots/{st.stats.starttime.strftime('%Y')}"
    os.makedirs(name=output_dir, exist_ok=True)
    src_file = f"{output_dir}/" \
               f"{str(st.stats.starttime.julday).zfill(3)}_{st.stats.starttime.strftime('%Y-%m-%dT%H:%M:%S')}.png"
    plt.savefig(src_file, dpi=600, transparent=True)
    plt.show()
    plt.close(fig)

    if np.any(pro_mean > threshold):
        output_dir = f"{output_dir}/selected"
        os.makedirs(name=output_dir, exist_ok=True)
        dst_file = f"{output_dir}/" \
                   f"{str(st.stats.starttime.julday).zfill(3)}_{st.stats.starttime.strftime('%Y-%m-%dT%H:%M:%S')}.png"
        shutil.copy2(src_file, dst_file)

        id = np.where(pro_mean > threshold)[0]
        t1 = (st.stats.starttime + 60 * id[0]).strftime("%Y-%m-%dT%H:%M:%S")
        t2 = (st.stats.starttime + 60 * id[-1]).strftime("%Y-%m-%dT%H:%M:%S")
        duration = st.stats.starttime + 60 * id[-1] - st.stats.starttime + 60 * id[0]
        margin_error = np.mean(pro_95ci_range[id])

        record = f"{t1}, {t2}, {duration :.2f}, {np.sum(pro_mean[id]) :.2f}, {margin_error :.2f}, {threshold}"
        with open(f"{output_dir}/{st.stats.starttime.strftime('%Y')}_selected.txt", "a") as f:
            f.write(record + "\n")

def workflow(year, julday):

    catchment_name = "Luding"

    if year == 2023:
        seismic_network, input_station, input_component = "WD", "STA02", "BHZ"
    else:
        seismic_network, input_station, input_component = "AM", "R9BF5", "EHZ"


    # predicted results
    df = pd.read_csv(f"{current_dir}/"
                     f"{catchment_name}-{seismic_network}-{year}-{input_station}-{input_component}-E-testing-False-predicted.txt",
                     header=0)
    df_date = np.array(df.iloc[:, 0])
    df_arr = np.array(df.iloc[:, 1:]).astype(float)


    # seismic data
    data_start = UTCDateTime(year=year, julday=julday)
    data_end = UTCDateTime(year=year, julday=julday + 1)
    st = load_seismic_signal(catchment_name, seismic_network, input_station, input_component, data_start, data_end, f_min=1, f_max=50)


    for hour in range(0, 21, 4):  # like 0-4, 4-8, ..., 20-24
        tr = st.copy()

        if hour == 20:
            tr.trim(UTCDateTime(year=year, julday=julday, hour=hour),
                    UTCDateTime(year=year, julday=julday, hour=23, minute=55))
        else:
            tr.trim(UTCDateTime(year=year, julday=julday, hour=hour),
                    UTCDateTime(year=year, julday=julday, hour=hour + 4))

        tr = convert_st2tr(tr)

        id1 = np.where(df_date == tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S"))[0][0]
        id2 = np.where(df_date == tr.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S"))[0][0]

        pro_mean = df_arr[id1:id2, -2]
        pro_95ci_range = df_arr[id1:id2, -1]

        plot_3_subplot(pro_mean, pro_95ci_range, st=tr)

def main(year):

    logging.basicConfig(
        filename=f"{current_dir}/logs/job_{year}.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    for julday in range(157, 307):

        try:
            workflow(year, julday)
            logging.info(f"done {julday}")
        except Exception as e:
            print(f"error {julday}: {e}")
            logging.info(f"error {julday}: {e}")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--year", type=int)

    args = parser.parse_args()

    main(args.year)
