#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime

from results_visualization import *

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir

import torch
import torch.nn as nn


class lstm_classifier(nn.Module):
    def __init__(self, feature_size, device, dropout=0.25, num_hidden=512, num_layers=4, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=num_hidden,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False)

        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.fully_connect = nn.Linear(num_hidden, output_dim)
        self.device = device

    def forward(self, x):
        self.lstm.flatten_parameters()
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0)) # do not need (h_n, c_n)
        # x.shape = [batchSize, sequenceLEN, featureSIZE]
        # out.shape = [batchSize, sequenceLEN, num_hidden]
        # h_n.shape = c_n.shape = [num_layers, batchSize, num_hidden]
        out = out[:, -1, :]
        out = self.fully_connect(out)

        return out


class lstm_train_test:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 device: str,

                 output_dir: str,

                 input_station: str,
                 model_type: str,
                 feature_type: str,
                 input_component: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler
                 ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

        # give second output_logit more weight [0.01noise, 0.99DF]
        class_weight = torch.tensor([0.1, 0.9]).to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weight)

        self.min_loss = 1
        self.test_f1 = 0

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.input_station = input_station
        self.model_type = model_type
        self.feature_type = feature_type
        self.input_component = input_component
        self.scheduler = scheduler

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()

        path = f"{self.output_dir}/ckp"
        os.makedirs(name=path, exist_ok=True)

        torch.save(ckp, f"{path}/{self.input_station}_{self.model_type}_{self.feature_type}_{self.input_component}.pt")
        print(f"Model saved at epoch {epoch}, {path}", flush=True)

    def _save_output(self, be_saved_tensor, training_or_testing):

        be_saved_tensor = be_saved_tensor.detach().cpu().numpy()
        be_saved_tensor = be_saved_tensor[be_saved_tensor[:, 0].argsort()] # make sure the first column is time stamps
        be_saved_tensor[:, 3] = np.round(be_saved_tensor[:, 3], 3) # make sure the last column is predicted probability


        time_stamps_float = be_saved_tensor[:, 0]
        time_stamps_string = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S') for ts in time_stamps_float]
        time_stamps_string = np.array(time_stamps_string).reshape(-1, 1)


        save_output = np.hstack((time_stamps_string, be_saved_tensor[:, 1:])) # do not save the float time stamps


        output_path = f"{self.output_dir}/predicted_results/"
        os.makedirs(output_path, exist_ok=True)
        np.savetxt(f"{output_path}"
                   f"{self.input_station}_{self.model_type}_{self.feature_type}_{self.input_component}_{training_or_testing}_output.txt",
                   save_output, delimiter=',', fmt='%s', comments='',
                   header="time_window_start,obs_y_label,pre_y_label,pre_y_pro")

        return save_output

    def training(self, epoch):
        self.model.train()
        tensor_temp = torch.empty((0, 4)).to(self.device)
        epoch_loss = 0

        for batch_data in self.train_dataloader:
            input = batch_data['features'].to(self.device)  # Shape: (batch_size, sequence_length, feature_size)
            target = batch_data['label'].to(self.device)    # Shape: (sequence_length, 1) 0:NoneDF, 1:DF

            output_logit = self.model(input)  # return the model output logits, Shape (batch_size, 2)
            loss = self.loss_func(output_logit, target)
            epoch_loss += loss.item()

            # update the gredient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # prepare data
            time_stamps = batch_data['timestamps'].to(self.device)
            obs_y_label = target
            # predicted probability of debris flow label 1, Shape (batch_size, 1)
            pre_y_label = torch.argmax(output_logit, dim=1)
            # predicted probability, Shape (batch_size, 2)
            pre_y_pro = torch.softmax(output_logit, dim=1)[:, 1]  # selected the debris flow probability Shape (batch_size, 1)


            record = torch.cat((time_stamps.view(-1, 1),
                                obs_y_label.view(-1, 1),
                                pre_y_label.view(-1, 1),
                                pre_y_pro.view(-1, 1)), dim=1)
            tensor_temp = torch.cat((tensor_temp, record), dim=0)

        epoch_loss /= len(self.train_dataloader)

        print(f"Training at {epoch}, "
              f"{self.input_station}, {self.model_type}, {self.feature_type}, {self.input_component}, "
              f"epoch_loss, {epoch_loss}")

        # save the model if loss decrease
        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            # save the model
            self._save_checkpoint(epoch)
            # save train results
            self._save_output(tensor_temp, "training")

            visualize_confusion_matrix(self.output_dir, tensor_temp[:, 1].detach().cpu().numpy(),
                                       tensor_temp[:, 2].detach().cpu().numpy(), "training",
                                       self.input_station, self.model_type, self.feature_type, self.input_component)


    def testing(self, epoch):
        self.model.eval()
        val_loss = 0
        # loop all testing data
        with torch.no_grad():
            tensor_temp = torch.empty((0, 4)).to(self.device)

            for batch_data in self.test_dataloader:
                input = batch_data['features'].to(self.device)  # Shape: (batch_size, sequence_length, feature_size)
                target = batch_data['label'].to(self.device)  # Shape: (sequence_length, 1) 0:NoneDF, 1:DF

                output_logit = self.model(input)  # return the model output logits, Shape (batch_size, 2)
                loss = self.loss_func(output_logit, target)
                val_loss += loss.item()

                # prepare data
                time_stamps = batch_data['timestamps'].to(self.device)
                obs_y_label = target
                # predicted probability of debris flow label 1, Shape (batch_size, 1)
                pre_y_label = torch.argmax(output_logit, dim=1)
                # predicted probability, Shape (batch_size, 2)
                pre_y_pro = torch.softmax(output_logit, dim=1)[:,1]  # selected the debris flow probability Shape (batch_size, 1)


                record = torch.cat((time_stamps.view(-1, 1),
                                    obs_y_label.view(-1, 1),
                                    pre_y_label.view(-1, 1),
                                    pre_y_pro.view(-1, 1)), dim=1)
                tensor_temp = torch.cat((tensor_temp, record), dim=0)

            val_loss /= len(self.test_dataloader)

            f1 = f1_score(tensor_temp[:, 1].detach().cpu().numpy(),
                          tensor_temp[:, 2].detach().cpu().numpy(), average='binary', zero_division=0)

            if f1 > self.test_f1:
                self.test_f1 = f1

                self._save_output(tensor_temp, "testing")

                visualize_confusion_matrix(self.output_dir, tensor_temp[:, 1].detach().cpu().numpy(),
                                           tensor_temp[:, 2].detach().cpu().numpy(), "testing",
                                           self.input_station, self.model_type, self.feature_type, self.input_component)
                print(f"Testing at {epoch}, "
                      f"{self.input_station}, {self.model_type}, {self.feature_type}, {self.input_component}, "
                      f"F1, {f1}")
            else:
                print(f"Nothing saved, Testing at {epoch}, f1={f1} < self.test_f1={self.test_f1}")

        return val_loss


    def dual_testing(self):
        self.model.eval()
        val_loss = 0
        # loop all testing data
        with torch.no_grad():
            tensor_temp = torch.empty((0, 4)).to(self.device)

            for batch_data in self.test_dataloader:
                input = batch_data['features'].to(self.device)  # Shape: (batch_size, sequence_length, feature_size)
                target = batch_data['label'].to(self.device)  # Shape: (sequence_length, 1) 0:NoneDF, 1:DF

                output_logit = self.model(input)  # return the model output logits, Shape (batch_size, 2)
                loss = self.loss_func(output_logit, target)
                val_loss += loss.item()

                # prepare data
                time_stamps = batch_data['timestamps'].to(self.device)
                obs_y_label = target
                # predicted probability of debris flow label 1, Shape (batch_size, 1)
                pre_y_label = torch.argmax(output_logit, dim=1)
                # predicted probability, Shape (batch_size, 2)
                pre_y_pro = torch.softmax(output_logit, dim=1)[:,1]  # selected the debris flow probability Shape (batch_size, 1)


                record = torch.cat((time_stamps.view(-1, 1),
                                    obs_y_label.view(-1, 1),
                                    pre_y_label.view(-1, 1),
                                    pre_y_pro.view(-1, 1)), dim=1)
                tensor_temp = torch.cat((tensor_temp, record), dim=0)

            val_loss /= len(self.test_dataloader)

            self._save_output(tensor_temp, "dual_testing")


    def activation(self, num_epoch=100):

        for epoch in range(num_epoch): # loop 50 times for training
            self.training(epoch) # train the model every epoch

            val_loss = self.testing(epoch)

            self.scheduler.step(val_loss)

