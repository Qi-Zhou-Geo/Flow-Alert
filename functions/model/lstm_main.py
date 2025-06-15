#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

from datetime import datetime

# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
import torch.optim as optim
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.public.load_data import select_features
from functions.public.dataset_to_dataloader import *
from functions.public.min_max_normalize_transformer import min_max_normalize
from functions.model.lstm_model import LSTM_Classifier
from functions.model.train_test import Train_Test
from functions.public.undersamp_training_data import *

def prepare_dataloader(feature_type, batch_size, seq_length, noise2event_ratio, params, repeate=1):

    # empty list to store the dataloader
    train_sequences = []
    test_sequences = []
    validate_sequences = []

    # loop the input data-60s
    for p in params:

        seismic_network, input_year, input_station, input_component, dataloader_type, with_label = p.split("-")

        # convert str to bool
        if with_label == "True":
            with_label = True
        else:
            with_label = False

        # load data_array as [time_stamps, features, target]
        catchment_name = "Illgraben"
        input_features_name, data_array = select_features(catchment_name,
                                                          seismic_network,
                                                          input_year,
                                                          input_station,
                                                          input_component,
                                                          feature_type,
                                                          with_label,
                                                          repeate=repeate,
                                                          normalize=True)

        # convert data-60s frame to data-60s loader
        sequences = data_to_seq(array=data_array, seq_length=seq_length)

        # add sequences to list
        if dataloader_type == "training":
            train_sequences = train_sequences + sequences
        elif dataloader_type == "testing":
            test_sequences= test_sequences + sequences
        elif dataloader_type == "validation":
            validate_sequences = validate_sequences + sequences
        else:
            print(f"check the dataloader_type={dataloader_type}")

    # sample the sequences for training
    if noise2event_ratio < 300:
        # sample the training data-60s, and make sure "Non-DF : DF = noise2event_ratio : 1"
        train_sequences = under_sample_seq(train_sequences, noise2event_ratio)
    else:
        # use all training data-60s
        pass

    # convert sequences to data-60s loader
    train_dataloader = []
    test_dataloader = []
    validate_dataloader = []

    # train data-60s laoder, len(train_sequences) == 1
    dataset = seq_to_dataset(sequences=train_sequences, data_type="feature")
    dataloader = dataset_to_dataloader(dataset=dataset,
                                       batch_size=batch_size,
                                       training_or_testing="training")
    train_dataloader.append(dataloader.dataLoader())

    # test data-60s laoder, len(test_sequences) == 1
    dataset = seq_to_dataset(sequences=test_sequences, data_type="feature")
    dataloader = dataset_to_dataloader(dataset=dataset,
                                       batch_size=batch_size,
                                       training_or_testing="testing")
    test_dataloader.append(dataloader.dataLoader())

    # check the data-60s loader length
    if len(train_dataloader) == 0:
        # give a "fake" dataloader to make sure lstm_train_test works well
        train_dataloader.append(test_dataloader[0])
    else:
        pass

    if len(test_dataloader) == 0:
        # give a "fake" dataloader to make sure lstm_train_test works well
        test_dataloader.append(train_dataloader[0])
    else:
        pass

    if len(validate_dataloader) == 0:
        # give a "fake" dataloader to make sure lstm_train_test works well
        validate_dataloader.append(train_dataloader[0])
    else:
        pass

    return train_dataloader, test_dataloader, validate_dataloader

def load_model(feature_type, batch_size, seq_length, training_or_testing, device):

    map_feature_size = {"A":11, "B":69, "C":80, "D":70, "E":5}
    if feature_type in ["A", "B", "C", "D", "E"]:
        lstm_feature_size = map_feature_size.get(feature_type)
    else:
        print(feature_type)
        lstm_feature_size = int(feature_type.split("-")[1]) # as 'R-60-LSTM'

    model = LSTM_Classifier(feature_size=lstm_feature_size, device=device)

    if training_or_testing == "training":
        pass
    elif training_or_testing == "testing":
        # the default reference model is trained with "2017-2019 9S-ILL12" data-60s
        try:
            load_ckp = f"{CONFIG_dir['ref_model_dir']}/ref-train-9S-2017_2019-ILL12-EHZ-{feature_type}-att.pt"
        except Exception as e:
            sys.exit(f"Exiting the process due to an error, {e}")
        model.load_state_dict(torch.load(load_ckp, map_location=torch.device('cpu')))
    else:
        print(f"load model failed, p={feature_type, training_or_testing, device}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Define scheduler: Reduce the LR by factor of 0.1 when the metric (like loss) stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # print the model structure
    summary(model=model,
            input_size=[(batch_size, seq_length, lstm_feature_size), (batch_size, seq_length)],
            col_names=("input_size", "output_size", "num_params", "params_percent", "trainable"),
            device=device)

    return model, optimizer, scheduler


def main(model_type, feature_type, batch_size, seq_length,
         class_weight, noise2event_ratio, params, repeate,
         output_dir=None):

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"Start Job={job_id}, UTC+0={time_now}, "
          f"{model_type, feature_type, batch_size, seq_length, class_weight, noise2event_ratio, repeate}", "\n")

    # working device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data-60s
    train_dataloader, test_dataloader, validate_dataloader = prepare_dataloader(feature_type,
                                                                                batch_size,
                                                                                seq_length,
                                                                                noise2event_ratio,
                                                                                params)
    # data-60s information from params
    seismic_network, _, input_station, input_component, training_or_testing, _  = params[0].split("-")

    # load model
    model, optimizer, scheduler = load_model(feature_type, batch_size, seq_length, training_or_testing, device)

    # train or test class
    input_format = f"{params[0]}-repeate-{repeate}-{model_type}-{feature_type}-DFweight-{class_weight}-ratio-{noise2event_ratio}"

    if output_dir is not None:
        output_dir = output_dir
    else:
        output_dir = f"{CONFIG_dir['output_dir']}"

    workflow = Train_Test(model, optimizer, scheduler,
                          train_dataloader, test_dataloader, validate_dataloader,
                          device, output_dir, input_format, model_type,
                          class_weight, noise2event_ratio, data_type="feature")


    if training_or_testing == "training":
        workflow.activation()
    elif training_or_testing == "testing":
        workflow.testing(received_dataloader=test_dataloader)
    else:
        print(f"checke the training_or_testing")

    time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"End Job={job_id}: UTC+0={time_now}, "
          f"{model_type, feature_type, batch_size, seq_length, class_weight, noise2event_ratio, repeate}", "\n")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--model_type", default="LSTM", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")

    parser.add_argument("--batch_size", default=16, type=int, help='input batch size on each device')
    parser.add_argument("--seq_length", default=32, type=int, help="input sequence length")

    parser.add_argument("--class_weight", default=0.9, type=float, help="weight for DF label")
    parser.add_argument("--noise2event_ratio", default=1, type=int, help="Non-DF to DF label ratio")

    parser.add_argument("--params", nargs='+', type=str, help="list of stations")

    parser.add_argument("--num_repeate", default=6, type=int, help="num of repeate")
    parser.add_argument("--output_dir", default="CONFIG_dir['output_dir_div']", type=str, help="model output dir")

    args = parser.parse_args()

    for repeate in range(1, args.num_repeate + 1):  # repate 5 times
        main(args.model_type, args.feature_type,
             args.batch_size, args.seq_length,
             args.class_weight, args.noise2event_ratio,
             args.params, repeate, args.output_dir)
