#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import sys
import yaml

import math
import numpy as np
from scipy.stats import t as student_t  # Student's t-distribution

import torch
import torch.nn as nn
# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0

from pathlib import Path
from tqdm import tqdm


class LSTM_Classifier(nn.Module):
    def __init__(self, feature_size, device,
                 hidden_size=512, num_layers=4,
                 dropout=0.1, bidirectional=False, output_dim=2):
        super().__init__()

        # suppose input
        # "t" is time stamps of t to t_i
        # shape "t" = ([batch_size, sequence_length])
        # "x" is features of t to t_i
        # shape "x" = ([batch_size, sequence_length, feature_size])

        # "lstm" receive "x", and return "y" ([batch_size, output_dim])

        # initial parameters
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.output_dim = output_dim

        # same as "D = 2 if bidirectional=True otherwise 1" in Pytorch
        if self.bidirectional is True:
            self.D = 2
        else:
            self.D = 1

        # lstm layer
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional)

        # fully connected layer
        self.fully_connect = nn.Linear(self.hidden_size * self.D, self.output_dim)

    def forward(self, x, t=None):

        x = x.to(torch.float32)
        self.lstm.flatten_parameters()

        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size, dtype=x.dtype).to(self.device)
        c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size, dtype=x.dtype).to(self.device)

        # output.shape = ([batch_size, sequence_length, hidden_size])
        # h_n.shape = c_n.shape = ([num_layers * self.D, batch_size, hidden_size])
        output, (h_n, c_n) = self.lstm(x, (h0, c0)) # do not need (h_n, c_n)


        # extract features
        if self.bidirectional is True:
            forward_last = h_n[-2, :, :]   # last layer's forward hidden state
            backward_last = h_n[-1, :, :]  # last layer's backward hidden state
            # concatenate along the feature dimension
            output = torch.cat((forward_last, backward_last), dim=1)
        else:
            #output = h_n[-1, :, :]
            # or use "output = output[:, -1, :]" to only focous the last time step in "sequence_length" domain
            output = output[:, -1, :]

        output = self.fully_connect(output)

        return output

class Ensemble_Trained_LSTM_Classifier:
    def __init__(self, trained_model_name, model_version, device, model_dir=None, station=None):

        self.trained_model_name = trained_model_name
        self.model_version = model_version
        self.device = device
        self.model_dir = model_dir
        self.station = station


    def load_trained_model(self, trained_model_name, repeate, attention=False, print_model_summary=False):

        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent

        config_path = f"{project_root}/config/config_inference.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_params = config[f"{trained_model_name}"]
        lstm_feature_size = model_params[f"lstm_feature_size"]
        ref_model_name = model_params[f"ref_model_name"]

        if self.station is None:
            pass
        else:
            ref_model_name = ref_model_name.replace("station", self.station)

        ref_model_name = ref_model_name.replace("X", str(repeate))
        batch_size = model_params[f"batch_size"]
        seq_length = model_params[f"seq_length"]


        if attention is True:
            model = LSTM_Attention(feature_size=lstm_feature_size, device=self.device)
        else:
            model = LSTM_Classifier(feature_size=lstm_feature_size, device=self.device)

        if self.model_dir is None:
            ref_model_dir = f"{project_root}/trained_model/{self.model_version}"
        else:
            ref_model_dir = self.model_dir

        load_checkpoint = torch.load(f"{ref_model_dir}/{ref_model_name}", map_location=torch.device('cpu'))
        model.load_state_dict(load_checkpoint)
        model.to(self.device)
        model.eval()  # set as "evaluate" mode

        if print_model_summary is True:
            s = summary(model=model,
                        input_size=[(batch_size, seq_length, lstm_feature_size), (batch_size, seq_length)],
                        col_names=("input_size", "output_size", "num_params", "params_percent", "trainable"),
                        device=self.device)

            print(f"Loaded pre-trained model ({trained_model_name}) successfully.\n"
                  f"Please check the model details below:\n"
                  f"{s}")

        return model

    def ensemble_models(self, num_repeate=5, attention=False, print_model_summary=False):

        models = []
        for repeate in range(1, num_repeate+1):
            model = self.load_trained_model(trained_model_name=self.trained_model_name,
                                            repeate=repeate,
                                            attention=attention,
                                            print_model_summary=print_model_summary)
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

        predicted_pro = []
        for model in models:
            # make sure does not change the model parameters
            model.eval()
            with torch.no_grad():
                logits = model(features, t_features)
                DF_pro = torch.softmax(logits, dim=1)[:, 1]
                DF_pro = DF_pro.cpu().detach().numpy()
                predicted_pro.append(DF_pro)

        predicted_pro = [arr[0].item() for arr in predicted_pro]
        pro_mean, ci_range = self.statistical_testing(predicted_pro)
        predicted_pro = [float(f"{i:.3f}") for i in predicted_pro]

        return predicted_pro, pro_mean, ci_range

class LSTM_Attention(nn.Module):
    def __init__(self, feature_size, device,
                 hidden_size=256, num_layers=4,
                 dropout=0.25, bidirectional=False, output_dim=2):
        super().__init__()

        # Keep original structure
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        # same as "D = 2 if bidirectional=True otherwise 1" in Pytorch
        if bidirectional is True:
            self.D = 2
        else:
            self.D = 1

        # lstm layer
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # set forget gate bias to 1 to help memory retention
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)


        # attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.D,
            num_heads=min(8, num_layers),
            dropout=dropout,
            batch_first=True)

        # layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.D)

        # learnable attention pooling parameters
        self.attn_pool_weight = nn.Parameter(torch.Tensor(hidden_size * self.D, 1))
        nn.init.xavier_uniform_(self.attn_pool_weight)

        # output layer
        self.fully_connect = nn.Sequential(
            nn.Linear(hidden_size * self.D, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )
        nn.init.xavier_uniform_(self.fully_connect[0].weight)
        nn.init.xavier_uniform_(self.fully_connect[-1].weight)

    def _normalize(self, x, epsilon=1e-6):
        # shape of x is ([batch_size, sequence_length, num_stations * num_channels = feature size])

        #seq_mean = x.mean(dim=1, keepdim=True)
        #seq_std = x.std(dim=1, keepdim=True)
        #x = (x - seq_mean) / (seq_std + epsilon)

        median = x.median(dim=1, keepdim=True).values
        q1 = x.quantile(0.25, dim=1, keepdim=True)
        q3 = x.quantile(0.75, dim=1, keepdim=True)
        x = (x - median) / (q3 - q1 + epsilon)

        return x

    def _pos_encoder(self, x):

        sequence_length, feature_size = x.shape[1], x.shape[2]

        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, feature_size, 2) * -(np.log(10000.0) / feature_size))
        pe = np.zeros((sequence_length, feature_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:(feature_size // 2)])

        pe_matrix = torch.tensor(pe, dtype=x.dtype).to(self.device)
        pe_std = x.std(dim=(0, 1), keepdim=True)  # Compute the standard deviation across batch & time
        pe_matrix = pe_matrix * pe_std  # Scale positional encoding
        # add positional encoding along the sequence_length dimension
        x = x + pe_matrix

        return x


    def forward(self, x, t=None, hidden=None):
        x = x.to(torch.float32)
        #x = self._normalize(x)
        #x = self._pos_encoder(x)

        self.lstm.flatten_parameters()

        # Initialize hidden states (keep your original logic)
        if hidden is None:
            h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size, dtype=x.dtype).to(self.device)
            c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size, dtype=x.dtype).to(self.device)
        else:
            h0, c0 = hidden
            h0, c0 = h0.to(self.device), c0.to(self.device)

        # output.shape = ([batch_size, sequence_length, hidden_size])
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        # layer normalization
        output = self.layer_norm(output)

        # attention with residual connection
        attn_output, _ = self.attention(output, output, output)
        output = output + attn_output  # skip connection helps gradient flow

        # pooling and output
        #pooled_output = torch.mean(output, dim=1)
        attn_scores = torch.matmul(output, self.attn_pool_weight)  # Shape: (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # normalize scores
        pooled_output = torch.sum(output * attn_weights, dim=1)  # weighted sum across time steps

        output = self.fully_connect(pooled_output)

        return output
