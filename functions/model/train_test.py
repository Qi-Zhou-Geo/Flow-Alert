#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import logging

from typing import List

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datetime import datetime


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
#from functions.model.results_visualization import *
from functions.public.archive_data import *

def _save_checkpoint(model, output_dir, output_format, epoch):

    ckp = model.state_dict()
    path = f"{output_dir}/{output_format}.pt"
    torch.save(ckp, path)

    print(f"Model saved at epoch {epoch}, {path}", flush=True)

def _plot(tensor_temp, purpose, output_dir, output_format):

    if purpose != "training":
        # probability map
        try:
            visualize_probability_map(tensor_temp[:, 0],
                                      tensor_temp[:, 1],
                                      tensor_temp[:, 3],
                                      purpose,
                                      output_dir,
                                      output_format)
        except Exception as e:
            print(f"{purpose}, {output_format}, {e} in visualize_probability_map")
    else:
        pass

    # confusion matrix
    try:
        visualize_confusion_matrix(tensor_temp[:, 2],
                                   tensor_temp[:, 4],
                                   purpose,
                                   output_dir,
                                   output_format)
    except Exception as e:
        print(f"{purpose}, {output_format}, {e} in visualize_confusion_matrix")


class Train_Test:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,

                 train_dataloader: List[DataLoader],
                 test_dataloader: List[DataLoader],
                 validate_dataloader: List[DataLoader],

                 device: str,

                 # input_format
                 output_dir: str = None,
                 input_format: str = "9S-2017_2019-ILL12-EHZ-C",
                 model_type: str = "results",

                 # Non-DF label : DF label = 1-class_weight : class_weight
                 class_weight: float = 0.9,  # weight for DF label
                 noise2event_ratio: int = 1e5, # either sample the training data-60s or use all training data-60s

                 data_type: str = "feature",
                 ) -> None:

        # <editor-fold desc="set parameter that used in class">
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.validate_dataloader = validate_dataloader

        self.device = device
        self.model_type = model_type
        # </editor-fold>


        # <editor-fold desc="set class output dir and format">
        if output_dir is None:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            output_dir = f"{project_root}/output"

        self.output_dir = f"{output_dir}/{model_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_format = input_format
        # </editor-fold>


        # <editor-fold desc="set initial monitor parameter value">
        # be carfule the nagative symbol "-"
        self.train_monitor = 1e3 # monitor epoch loss
        self.test_monitor = -1e3 # monitor F1
        self.validate_monitor = -1e3 # monitor F1
        # </editor-fold>


        # <editor-fold desc="set loss func">
        if data_type == "feature":
            class_weight_binary = torch.tensor([1-class_weight, class_weight]).to(self.device)
        elif data_type == "waveform":
            class_weight_binary = torch.tensor([1-class_weight, class_weight]).to(self.device)# [0.004, 0.996]
        else:
            print(f"check the training/testing data type, data_type={data_type}")
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weight_binary)
        # </editor-fold>

    def training(self, epoch, dump_model=True, purpose="training"):
        self.model.train()
        tensor_temp = torch.empty((0, 5)).to(self.device)
        epoch_loss, dataloader_leangth = 0, 0

        # first loop all dataloader
        for dataloader in self.train_dataloader:
            # second loop all batch
            dataloader_leangth += len(dataloader)
            for batch_data in dataloader:
                # t_features of t to t_{sequence_length}, shape ([batch_size, sequence_length]), float time stamps
                t_features = batch_data['t_features'].to(self.device)
                # features of t to t_i, shape ([batch_size, sequence_length, num_stations * num_channels])
                features = batch_data['features'].to(self.device)

                # t_target of t_{sequence_length+1}, shape ([batch_size, 1]), float time stamps
                t_target = batch_data['t_target'].to(self.device)
                # target of t_{sequence_length+1}, shape ([batch_size, 1]), debris flow probability or label
                target = batch_data['target'].to(self.device)

                predicted_pro = self.model(features, t_features)  # return the model output logits, shape (batch_size, 2)
                loss = self.loss_func(predicted_pro, target)
                epoch_loss += loss.item()

                # update the gredient
                self.optimizer.zero_grad()
                loss.backward()
                # clip the grad to avoid the explosion
                if self.model_type in ['xLSTM', 'sLSTM', 'mLSTM']:
                    pass
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=10,
                        norm_type=2
                    )
                    if total_norm > 10:
                        print(f"Clipped from {total_norm:.2f} to 1.0")

                self.optimizer.step()

                # predicted probability of debris flow label 1
                obs_y_label = (target >= 0.5).float()
                predicted_pro = torch.softmax(predicted_pro, dim=1)[:, 1]
                pre_y_label = (predicted_pro >= 0.5).float()
                record = torch.cat((t_target.view(-1, 1),
                                    target.view(-1, 1),
                                    obs_y_label.view(-1, 1),
                                    predicted_pro.view(-1, 1),
                                    pre_y_label.view(-1, 1)), dim=1) # as column
                tensor_temp = torch.cat((tensor_temp, record), dim=0) # as row

        epoch_loss /= dataloader_leangth

        # make sure the first column is time stamps
        tensor_temp = tensor_temp.detach().cpu().numpy()
        tensor_temp = tensor_temp[tensor_temp[:, 0].argsort()]

        # dump the results
        be_saved_array = tensor_temp
        evaluate_matrix = dump_evaluate_matrix(be_saved_array,
                                               f"{purpose}, epoch_{epoch}_{self.output_format}",
                                               f"{self.output_dir}", f"summary_{self.model_type}_all",
                                               dump=dump_model)
        # update the training loss
        if epoch_loss < self.train_monitor:
            self.train_monitor = epoch_loss
            # if epoch decrease, dump the training results
            dump_model_prediction(be_saved_array, purpose, self.output_dir, self.output_format)


        if dump_model is True:
            _save_checkpoint(self.model, self.output_dir, f"{purpose}-{self.output_format}-at-{epoch}", epoch)


        return epoch_loss, evaluate_matrix

    def testing(self, epoch, purpose, received_dataloader):
        self.model.eval()
        # loop all testing data-60s
        with torch.no_grad():
            # first loop all dataloader
            for idx, dataloader in enumerate(received_dataloader):
                tensor_temp = torch.empty((0, 5)).to(self.device)
                epoch_loss = 0
                # second loop all batch
                for batch_data in dataloader:
                    # t_features of t to t_{sequence_length}, shape ([batch_size, sequence_length]), float time stamps
                    t_features = batch_data['t_features'].to(self.device)
                    # features of t to t_i, shape ([batch_size, sequence_length, num_stations * num_channels])
                    features = batch_data['features'].to(self.device)

                    # t_target of t_{sequence_length+1}, shape ([batch_size, 1]), float time stamps
                    t_target = batch_data['t_target'].to(self.device)
                    # target of t_{sequence_length+1}, shape ([batch_size, 1]), debris flow probability or label
                    target = batch_data['target'].to(self.device)

                    predicted_pro = self.model(features, t_features)  # return the model output logits, shape (batch_size, 2)
                    loss = self.loss_func(predicted_pro, target)
                    epoch_loss += loss.item()

                    # predicted probability of debris flow label 1
                    obs_y_label = (target >= 0.5).float()
                    predicted_pro = torch.softmax(predicted_pro, dim=1)[:, 1]
                    pre_y_label = (predicted_pro >= 0.5).float()
                    record = torch.cat((t_target.view(-1, 1),
                                        target.view(-1, 1),
                                        obs_y_label.view(-1, 1),
                                        predicted_pro.view(-1, 1),
                                        pre_y_label.view(-1, 1)), dim=1)  # as column
                    tensor_temp = torch.cat((tensor_temp, record), dim=0)  # as row

                epoch_loss /= len(dataloader)

                # make sure the first column is time stamps
                tensor_temp = tensor_temp.detach().cpu().numpy()
                tensor_temp = tensor_temp[tensor_temp[:, 0].argsort()]

                # dump the results
                be_saved_array = tensor_temp
                evaluate_matrix = dump_evaluate_matrix(be_saved_array,
                                                       f"{purpose}, epoch_{epoch}_{self.output_format}",
                                                       f"{self.output_dir}", f"summary_{self.model_type}_all")

                if evaluate_matrix[-2] > self.test_monitor:
                    dump_model_prediction(be_saved_array, purpose, self.output_dir, self.output_format)

        return epoch_loss, evaluate_matrix

    def activation(self, num_epoch=50, testing=True, validation=False):

        # initial as none
        train_matrix_temp = ""
        test_matrix_temp = ""

        for epoch in range(1, num_epoch+1): # loop 50 times for training
            train_loss, train_matrix = self.training(epoch)  # train the model every epoch
            self.scheduler.step(train_loss)

            if testing is True:
                test_loss, test_matrix = self.testing(epoch, "testing", self.test_dataloader)
            else:
                test_loss, test_matrix = -1, [-1] * 8

            # check the test F1
            if test_matrix[-2] > self.test_monitor:
                self.test_monitor = test_matrix[-2] # update the monitor
                train_matrix_temp = f"training, epoch_{epoch}_{self.output_format}, {train_matrix} \n"
                test_matrix_temp = f"testing, epoch_{epoch}_{self.output_format}, {test_matrix} \n"
                _save_checkpoint(self.model, self.output_dir, f"{self.output_format}", epoch)

            # record the logs
            time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            record = f"epoch, {epoch}/{num_epoch}, UTC+0, {time_now}, train_loss, {train_loss}, test_loss, {test_loss}, " \
                     f"self.train_monitor, {self.train_monitor}, self.test_monitor, {self.test_monitor}, test_matrix, {test_matrix}\n"
            print(record)

        if validation is True :
            self.testing(num_epoch, "validation", self.validate_dataloader)

        # dump the optimal matrix to local
        dump_as_row(output_dir=f"{self.output_dir}",
                    output_name=f"summary_{self.model_type}_optimal",
                    variable_str=train_matrix_temp)
        dump_as_row(output_dir=f"{self.output_dir}",
                    output_name=f"summary_{self.model_type}_optimal",
                    variable_str=test_matrix_temp)
        