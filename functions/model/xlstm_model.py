#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-08-13
#__author__ = Kshitij Kar, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = kar@gfz.de, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import sys
import json
import numpy as np
from scipy.stats import t as student_t
from datetime import datetime
from tqdm import tqdm

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys
sys.path.append(str(project_root))
# </editor-fold>


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir
# print("PyTorch version:", torch.__version__) = PyTorch version: 2.6.0+cu118
import torch
import torch.nn as nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0

# import the custom functions
from functions.tool.copy_trained_model import rename_model

class xLSTM_Classifier(nn.Module):
    def __init__(self, feature_size, device,
                 conv1d_kernel_size=4, qkv_proj_blocksize=4, 
                 num_heads=4, context_length=64, 
                 num_blocks=2, hidden_size=256, 
                 slstm_at=[0], num_layers: int = 1,
                 dropout=0.25, output_dim=2):
        super(xLSTM_Classifier, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.output_dim = output_dim

        self.input_layer = nn.Linear(self.feature_size, self.hidden_size)

        backend = "vanilla"
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, 
                    qkv_proj_blocksize=qkv_proj_blocksize, 
                    num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=self.hidden_size,  # <-- since we're concatenating prev_targets
            slstm_at=slstm_at,
        )

        self.xlstm_layers = nn.ModuleList([xLSTMBlockStack(self.cfg) for _ in range(self.num_layers)])

        self.dropout_layer = nn.Dropout(self.dropout)
        self.fully_connected = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x, t=None):

        x = x.to(torch.float32)

        x = self.input_layer(x)
        for xlstm in self.xlstm_layers:
            x = xlstm(x)
        
        x = x[:, -1, :]

        x = self.dropout_layer(x)
        x = self.fully_connected(x)

        return x
    
class Ensemble_Trained_xLSTM_Classifier:
    def __init__(self, model_version, feature_type, batch_size, seq_length,
                 device, ML_name="xLSTM", station="ILL02"):

        self.model_version = model_version
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.seq_length = seq_length


        self.ML_name = ML_name
        self.station = station.replace("0", "1")

        self.device = device


    def load_trained_model(self, ML_name, station,
                           feature_type,
                           batch_size, seq_length,
                           repeat, print_model_summary=False):

        # <editor-fold desc="add the sys.path to search for custom modules">
        from pathlib import Path
        current_dir = Path(__file__).resolve().parent
        # using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
        project_root = current_dir.parent.parent

        import sys
        sys.path.append(str(project_root))
        # </editor-fold>

        feature_size, ref_model_name = rename_model(ML_name, station,
                                                    feature_type, batch_size, seq_length,
                                                    num_repeat=repeat)
        with open(f"{project_root}/config/xlstm_params.json", 'r') as f:
            xlstm_params = json.load(f)
        if feature_type in ['A', 'B', 'C']:
            model_params = xlstm_params.get(f"{ML_name.lower()}").get(f"{feature_type}").get(f"{station}")
        else:
            model_params = xlstm_params.get(f"{ML_name.lower()}").get(f"{feature_type}").get("default")
        model = xLSTM_Classifier(feature_size=feature_size, device=self.device, **model_params)

        ref_model_dir = f"{project_root}/trained_model/{self.model_version}"
        load_checkpoint = torch.load(f"{ref_model_dir}/{ref_model_name}", map_location=torch.device('cpu'))

        model.load_state_dict(load_checkpoint)
        model.to(self.device)
        model.eval()  # set as "evaluate" mode

        if print_model_summary is True and repeat == 1:
            s = summary(model=model,
                        input_size=[(batch_size, seq_length, feature_size), (batch_size, seq_length)],
                        col_names=("input_size", "output_size", "num_params", "params_percent", "trainable"),
                        device=self.device)

            print(f"Load pre-trained model \n"
                  f"<{ref_model_name}> \n"
                  f"from \n"
                  f"<{self.model_version}>\n"
                  f"successfully.\n \n"
                  f"Please check the model details below:\n"
                  f"{s}")

        return model

    def ensemble_models(self, num_repeat=5, print_model_summary=False):

        models = []
        for repeat in range(1, num_repeat + 1):

            model = self.load_trained_model(self.ML_name, self.station,
                                            self.feature_type,
                                            self.batch_size, self.seq_length,
                                            repeat,
                                            print_model_summary=print_model_summary)
            model.to(self.device)
            models.append(model)

        return models

    def statistical_testing(self, predicted_pro, row_or_column="column", confidence_level=0.95):
        # calculate the ranges based on confidence_level by student t distribution
        predicted_pro = np.array(predicted_pro)

        if predicted_pro.ndim == 1:
            axis = 0
        elif predicted_pro.ndim == 2:
            if row_or_column == "row":
                # calculate the mean and CI in the "row" dimension
                axis = 1
            else:
                # calculate the mean and CI in the "column" dimension
                axis = 0

        pro_mean = np.mean(predicted_pro, axis=axis)
        # standard Error of the Mean (pro_mean)
        sem = np.std(predicted_pro, axis=axis, ddof=1) / np.sqrt(predicted_pro.shape[axis])

        degree_of_freedom = predicted_pro.shape[axis] - 1
        alpha = 1 - confidence_level  # significance level
        # cumulative probability up to the critical value in the right tail of the distribution
        tail = 1 - alpha / 2
        ci_range = student_t.ppf(tail, degree_of_freedom) * sem  # by t-distribution

        return pro_mean, ci_range

    def predictor_from_dataLoader(self, dataloader, models):
        '''

        Args:
            dataloader: pytorch dataloader, with shape as ([time stamps, ])
            models: List[pytorch model]

        Returns:

        '''

        array_temp = np.empty((0, 4+len(models)))

        for batch_data in tqdm(dataloader,
                               desc="Progress of <predictor_from_dataLoader>",
                               file=sys.stdout):
            # t_features of t to t_{sequence_length}, shape ([batch_size, sequence_length]), float time stamps
            t_features = batch_data['t_features'].to(self.device)
            # features of t to t_i, shape ([batch_size, sequence_length, num_stations * num_channels])
            features = batch_data['features'].to(self.device)

            # t_target of t_{sequence_length+1}, shape ([batch_size, 1]), float time stamps
            t_target = batch_data['t_target']
            # target of t_{sequence_length+1}, shape ([batch_size, 1]), debris flow probability or label
            target = batch_data['target']

            predicted_pro = np.empty((len(target), len(models)))
            for idx, model in enumerate(models):
                # make sure does not change the model parameters
                model.eval()
                with torch.no_grad():
                    logits = model(features, t_features) # return the model output logits, shape (batch_size, 2)
                    DF_pro = torch.softmax(logits, dim=1)[:, 1]
                    DF_pro = DF_pro.cpu().detach().numpy()

                    predicted_pro[:, idx] = np.round(DF_pro.reshape(-1), 3) # keep 3 decimal places

            pro_mean, ci_range = self.statistical_testing(predicted_pro, row_or_column="row")

            record = np.concatenate((t_target.reshape(-1, 1),
                                     target.reshape(-1, 1),
                                     predicted_pro,
                                     pro_mean.reshape(-1, 1),
                                     ci_range.reshape(-1, 1)), axis=1)  # as column
            array_temp = np.vstack((array_temp, record))  # as row


        sort_indices = array_temp[:, 0].argsort()
        array_temp = array_temp[sort_indices]

        return array_temp

    def predictor_from_sequence(self, features, t_features, models):

        '''
        Ask the pre-trained model to inference

        Args:
            features: 2d numpy array, row as time step, column as features, shape as ([sequence length, feature size]);
            t_features: 1d numpy array, represents the time stamps
            models: List[model], pre-trained model
            data_type: str,

        Returns:
            model predicted debris flow probability, mean, 95% CI range
        '''

        features = features.reshape(1, features.shape[0], features.shape[1])
        features = features.astype(np.float32)
        features = torch.from_numpy(features)

        t_features = t_features.reshape(1, 1, -1)
        t_features = torch.from_numpy(t_features).unsqueeze(0)

        features, t_features = features.to(self.device), t_features.to(self.device)
        # print(features.size())
        # print(t_features.size())
        predicted_pro = []
        for model in models:
            # make sure does not change the model parameters
            model.eval()
            with torch.no_grad():
                logits = model(x= features, t= t_features)
                DF_pro = torch.softmax(logits, dim=1)[:, 1]
                DF_pro = DF_pro.cpu().detach().numpy()
                predicted_pro.append(DF_pro)

        predicted_pro = [arr[0].item() for arr in predicted_pro]
        pro_mean, ci_range = self.statistical_testing(predicted_pro)
        predicted_pro = [float(f"{i:.3f}") for i in predicted_pro]

        return predicted_pro, pro_mean, ci_range