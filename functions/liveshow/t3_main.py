#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-08T10:13:20
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import argparse

import time
import schedule

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# import the custom functions
from functions.toolkit.logger_printer import setup_logger

from functions.liveshow.t3s1_download_seismic import data_stream_pipeline, merge_seismic_data
from functions.liveshow.t3s2_run_model import run_flow_alert
from functions.liveshow.t3s4_plot_result import plot_daily

from functions.toolkit.send_email import usage

def run_pipeline(logger, remote_sub_folder, local_sub_folder):
    
    try:
        is_new_data, st = data_stream_pipeline(logger, remote_sub_folder, local_sub_folder)
        msg = f"<check_new_data_name> success"
    except Exception as e:
        is_new_data = False
        st = None
        msg = f"<check_new_data_name> fail:\n {e}"
    logger.info(msg)


    if is_new_data is True:
        # find new data
        try:
            run_flow_alert(st, output_path=f"{project_root}/deploy/liveshow_cache/pro")
        except Exception as e:
            msg = f"<run_flow_alert> failed:\n {e}"
            logger.info(msg)
    else:
        # no new data
        pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--log_output_dir", type=str, default=f"{project_root}/deploy/liveshow_cache/logs")
    parser.add_argument("--log_filename", type=str, default="t3_main.log")
    
    parser.add_argument("--remote_sub_folder", type=str, default="wsl_gfz")
    parser.add_argument("--local_sub_folder", type=str, default="deploy/liveshow_cache/seismic")
    args = parser.parse_args()
    
    # test the gmail connection
    usage()

    # setup logger
    logger = setup_logger(args.log_output_dir, args.log_filename, force_reset=False)
    
    # run it immediately
    run_pipeline(logger, args.remote_sub_folder, args.local_sub_folder)
    merge_seismic_data(logger, args.local_sub_folder)
    
    
    # repeat every 1 minutes
    schedule.every(2).minutes.do(run_pipeline, logger, args.remote_sub_folder, args.local_sub_folder)

    # repeat every 24 hours
    schedule.every().day.at("00:05").do(merge_seismic_data, logger, args.local_sub_folder)
    schedule.every().day.at("00:05").do(plot_daily)
    
    
    while True:
        schedule.run_pending()
        time.sleep(5) # sleep 5 seconds, then check schedule
        
        msg = f"<main> is sleeping.\n"
        logger.info(msg)