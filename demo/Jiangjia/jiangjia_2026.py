# !/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import os
import argparse

from scipy.stats import goodness_of_fit
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
import numpy as np
import pandas as pd
import torch.optim as optim
# from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0

from obspy import read, Trace, Stream, read_inventory, signal
from obspy.core import UTCDateTime

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.model.interface_model import FlowAlert

def main(model_type):

    st = read(f"{project_root}/demo/Jiangjia/JJG.453007897.EHZ.2025.210.disp.mseed")
    st.plot()
    st[0].stats.sampling_rate = 250
    # st = st.trim(UTCDateTime("2025-07-29T01:00:00"), UTCDateTime("2025-07-29T05:00:00"))

    model_version = "v1dot3model"
    flow_alert = FlowAlert(model_type, model_version, st, output_path=current_dir)
    flow_alert.model_config()
    flow_alert.model()
    flow_alert.feature()
    flow_alert.prediction()
    flow_alert.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="RF", type=str)
    args = parser.parse_args()

    main(args.model_type)

print()
