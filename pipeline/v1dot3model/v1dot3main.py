#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-12-14
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

from obspy import UTCDateTime

import torch
import torch.optim as optim
# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from train_test import Train_Test
from functions.data_process.df_to_dataloader import prepare_dataloader
from functions.model.lstm_model import LSTM_Attention


def load_model(feature_type, batch_size, seq_length, device):

    map_feature_size = {"A":11, "B":69, "C":80, "D":70, "E":13, "F":8, "G":10, "H":12}
    if feature_type in map_feature_size.keys():
        lstm_feature_size = map_feature_size.get(feature_type)
    else:
        print(feature_type)
        lstm_feature_size = int(feature_type.split("-")[1]) # as 'R-60-results'

    model = LSTM_Attention(feature_size=lstm_feature_size, device=device, dropout=0.25)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # Define scheduler: Reduce the LR by factor of 0.1 when the metric (like loss) stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # print the model structure
    summary(model=model,
            input_size=[(batch_size, seq_length, lstm_feature_size), (batch_size, seq_length)],
            col_names=("input_size", "output_size", "num_params", "params_percent", "trainable"),
            device=device)

    return model, optimizer, scheduler


def main(model_type, feature_type,
         batch_size, seq_length,
         params, repeat,
         output_dir=None, class_weight=0.9):

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    time_now = UTCDateTime().isoformat()
    print(f"Start Job={job_id}, UTC+0={time_now}, "
          f"{model_type, feature_type, batch_size, seq_length, repeat}", "\n")

    input_format = f"{params[0]}-repeat-{repeat}-{model_type}-{feature_type}-b{batch_size}-s{seq_length}"

    # working device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train_dataloader, test_dataloader, df = prepare_dataloader(params,
                                                               normalize=True,
                                                               batch_size=batch_size,
                                                               seq_length=seq_length)

    # save the input features
    input_feature = f"{output_dir}/LSTM/input_features_{feature_type}.txt"
    if os.path.isfile(input_feature):
        pass
    else:
        # for checking the data
        os.makedirs(f"{output_dir}/LSTM", exist_ok=True)
        df.to_csv(input_feature, index=False)

    # load model
    model, optimizer, scheduler = load_model(feature_type, batch_size, seq_length, device)

    # train or test class
    workflow = Train_Test(model=model, optimizer=optimizer, scheduler=scheduler,
                          train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                          device=device,
                          output_dir=output_dir, input_format=input_format,
                          model_type=model_type,
                          class_weight=class_weight)

    workflow.activation(num_epoch=50)

    time_now = UTCDateTime().isoformat()
    print(f"End Job={job_id}: UTC+0={time_now}, "
          f"{model_type, feature_type, batch_size, seq_length, repeat}", "\n")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--model_type", default="results", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")

    parser.add_argument("--batch_size", default=16, type=int, help='input batch size on each device')
    parser.add_argument("--seq_length", default=32, type=int, help="input sequence length")

    parser.add_argument("--params", nargs='+', type=str, help="list of stations")

    parser.add_argument("--num_repeat", default=5, type=int, help="num of repeat")
    parser.add_argument("--output_dir", default="output_dir_div", type=str, help="model output dir")

    args = parser.parse_args()

    for repeat in range(1, args.num_repeat + 1):  # repate 5 times
        main(args.model_type, args.feature_type,
             args.batch_size, args.seq_length,
             args.params, repeat, args.output_dir)