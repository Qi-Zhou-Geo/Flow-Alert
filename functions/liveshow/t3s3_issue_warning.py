#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-17T09:45:24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import json
import numpy as np
from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

from functions.toolkit.send_email import get_google_service, send_email_by_google


def warning_strategy(pro, attention_window=4, tolerance=0.5):
    
    mean_pro = np.mean(pro[-1 * attention_window :])
    
    if mean_pro >= tolerance:
        warning_status = True
    else:
        warning_status = False
    
    return warning_status



def recipient_list_level1():
    # "Recipient Level 1 (Real-time): you receive every warning update as it is issued."
    recipient_address = ["qi.zhou@gfz.de", "chow77@foxmail.com"]
    
    return recipient_address


def recipient_list_level2():
    # f"Recipient Level 2 (Summary): you receive only the first warning for each event.\n"
    # f"A new warning email will be sent only if a new event is detected after {x} hours."
    recipient_address = ["qi.zhou@gfz.de", "chow77@foxmail.com",
                         "kshitij.kar@gfz.de",
                         "kshitij797@gmail.com", "kshitij.kar@gfz.de", 
                         "hui.tang@gfz.de", "fabian.walter@wsl.ch"]
    
    return recipient_address


def set_recipient_level(duration=3600*2, key='Latest Warning [UTC+0]'):
    
    json_path = Path(project_root) / f"deploy/liveshow_cache/pro/last_Flow-Alert_warning.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # read the last warning time
    with open(json_path, "r") as f:
        last_warning = json.load(f)
        t1 = last_warning[key]
        t1 = UTCDateTime(t1)
    
    # check the time difference
    deltea_t = UTCDateTime().now() - t1
    if deltea_t < duration:
        recipient_level = "level1"
        recipient_note = "recipient_level 1 (Real-time): you receive every warning update as it is issued."
    else:
        recipient_level = "level2"
        recipient_note = (f"recipient_level 2 (Only-First-Warning): you receive only the first warning for each event." 
                          f"A new warning email will be sent only if a new event is detected after {duration/3600} hours.")
    
    # write the latest (or current warning)
    current_warning = {key: f"{UTCDateTime().now().isoformat()}"}
    with open(json_path, "w") as f:
        json.dump(current_warning, f)
    
    return recipient_level, recipient_note

def issue_warning(email_body, recipient_level="level1"):
    
    service = get_google_service()
    
    if recipient_level == "level1":
        recipient_address = recipient_list_level1()
    else:
        recipient_address = recipient_list_level2()
    
    for recipient in recipient_address:
        send_email_by_google(
            service,
            sender="qi.zhou.geo@gmail.com",
            to=recipient,
            subject="Illgraben Debris Flow Warning from Flow-Alert",
            body=email_body
            )