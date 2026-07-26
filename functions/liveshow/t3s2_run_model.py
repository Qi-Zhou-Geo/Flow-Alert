#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-26T11:47:13
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import json
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

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
from functions.model.interface_model import FlowAlert
from functions.liveshow.t3s3_issue_warning import warning_strategy, issue_warning, set_recipient_level


def run_flow_alert(st, output_path=None, sub_window_size=60, model_version="v1dot3model", model_type="LSTM"):

    # (1) load model and run it
    flow_alert = FlowAlert(model_type, model_version, st, 
                           output_path=output_path, 
                           sub_window_size=sub_window_size, 
                           window_overlap=0, 
                           clip_anomaly=True,
                           num_cpus=1)
    
    flow_alert.model_config()
    model_list = flow_alert.load_model()

    feature_arr = flow_alert.prepare_feature()
    model_output = flow_alert.make_prediction(tested_model=model_type)
    model_output[:, 0] = model_output[:, 0].astype(float) # float time stamps


    # (2) Add seismic feature RMSE
    feature_ts =  feature_arr[:, 0].astype(float)
    model_ts =  model_output[:, 0].astype(float)
    mask = np.isin(feature_ts, model_ts)
    
    rms = feature_arr[mask, 8] # RMS of the signal
    model_output = np.column_stack((model_output, rms.reshape(-1, 1)))

    # (3) write the last time stamps to txt
    last_row = model_output[-1, :]
    julday = UTCDateTime(last_row[1]).julday
    pro_path = Path(project_root) / f"deploy/liveshow_cache/pro/event_pro_{str(julday).zfill(3)}.txt" # type: ignore
    pro_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    num_model = len(model_list)
    header = ["time_stamps", "time_window_start"] + [f"pro{i}" for i in range(num_model)] + ["pro_mean", "pro_ci", "RMS"]
    if not pro_path.exists():
        # first run: write the full 2D array
        df = pd.DataFrame(model_output, columns=header)  
    else:
        # not the first run: append only the last row
        old_df = pd.read_csv(pro_path)
        new_df = pd.DataFrame(model_output, columns=old_df.columns)
        new_df["time_stamps"] = pd.to_numeric(new_df["time_stamps"])
        
        df = pd.concat([old_df, new_df], ignore_index=True, axis=0) # as row

    df = df.drop_duplicates(subset=["time_stamps"])
    df = df.sort_values(by="time_stamps").reset_index(drop=True)
    df.to_csv(pro_path, index=False)


    # (4) record the meta
    last_update = {f"Latest WSL 9S-ILL12-EHZ Data": f"{st[0].stats.endtime.strftime('%Y-%m-%dT%H:%M:%S')} [UTC+0]",
                   f"Latest Flow-Alert Update": f"{UTCDateTime().strftime('%Y-%m-%dT%H:%M:%S')} [UTC+0]"}
    json_path = Path(project_root) / f"deploy/liveshow_cache/pro/last_Flow-Alert_update.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(last_update, f)


    # (5) check warning status
    pro = model_output[:, -3].astype(float) # column "-3" is "pro_mean"
    warning_status = warning_strategy(pro, attention_window=4, tolerance=0.5)
    
    if warning_status is True:
        
        # (6-1) set email recipient
        recipient_level, recipient_note = set_recipient_level(duration=3600*2, key='Latest Warning [UTC+0]')
        
        # (6-2) set email details
        last_60 = model_output[-60:]
        model_output_str = "\t".join(header) + "\n"
        model_output_str = model_output_str + "\n".join(["\t".join(map(str, row)) for row in last_60])
        
        email_body = (f"This email was automatically sent from Flow-Alert v1.3.\n\n"
                      f"You are <recipient_level>: {recipient_level}.\n{recipient_note}"
                      f"{last_update}\n\n"
                      f"{model_output_str}")
        
        print(email_body)
        
        # (6-3) set email by Google
        issue_warning(email_body=email_body, recipient_level=recipient_level)
