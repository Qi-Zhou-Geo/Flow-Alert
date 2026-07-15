#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-15T16:51:21
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


import numpy as np

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


def recipient_list():
    
    recipient_address = ["qi.zhou@gfz.de", "qi.zhou.geo@gmail.com"
                         "kshitij797@gmail.com", "kshitij.kar@gfz.de", 
                         "hui.tang@gfz.de", "fabian.walter@wsl.ch"]
    
    return recipient_address

def issue_warning(email_body):
    
    service = get_google_service()
    recipient_address = recipient_list()
    
    for recipient in recipient_address:
        send_email_by_google(
            service,
            sender="qi.zhou.geo@gmail.com",
            to=recipient,
            subject="Illgraben Debris Flow Warning from Flow-Alert",
            body=email_body
            )