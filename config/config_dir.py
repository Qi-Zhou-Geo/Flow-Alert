#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-09-22
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import platform


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


def config_dir(project_root):
    sac_dir = f"/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic"
    output_dir = f"{project_root}/trained_model/slstmmodel_sl32_bs64"
    feature_output_dir = f"/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature"

    CONFIG_dir = {
        "sac_dir": sac_dir,
        "project_root": project_root,
        "output_dir": output_dir,
        "feature_output_dir":feature_output_dir,
    }

    return CONFIG_dir


def path_mapping(seismic_network):

    mapping = {"9J": "European/Illgraben", # EU data
               "9S": "European/Illgraben",
               }

    dir = mapping.get(seismic_network, "check function path_mapping")

    return dir



# please keep in mind this file I/O directory
CONFIG_dir = config_dir(project_root)

confidence_interval_to_z = {
    0.70: 1.04,
    0.75: 1.15,
    0.80: 1.28,
    0.85: 1.44,
    0.90: 1.645,
    0.92: 1.75,
    0.95: 1.96,
    0.96: 2.05,
    0.98: 2.33,
    0.99: 2.58
}
