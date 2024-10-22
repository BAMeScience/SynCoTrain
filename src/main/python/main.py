from __future__ import annotations

import argparse
import os

import pandas as pd

from src.main.python import configuration
from src.main.python.alignn import Alignn
from src.main.python.coTraining import CoTraining
from src.main.python.configuration import configure
from src.main.python.puLearning import PuLearning


def get_user_input():
    # helpful advices for users and get setup information
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction"
    )
    parser.add_argument(
        "--ehull015",
        type=bool,
        default=False,
        help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
    )
    parser.add_argument(
        "--small_data",
        type=bool,
        default=False,
        help="This option selects a small subset of data for checking the workflow faster.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=3,
        help="GPU ID to use for training.")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # use before loading lightning.gpu

    return args.ehull015, args.small_data


def get_data():
    # load dataframe
    df_path = configuration.input_dir + "/" + configuration.input_df_file
    return pd.read_pickle(df_path)


if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    ##########TEST##############
    test = configuration.current_setup(False, "alignn0", True)
    ############################

    # get user input
    ehull015, small_data = get_user_input()
    # read config file and set configurations
    config = configure(ehull015, small_data)
    # more setup needed?

    # read dataframe from file
    dataframe = get_data()

    # setup classifers and pu_learning for them
    alignn_one = Alignn()
    puLearning_one = PuLearning(alignn_one)
    puLearning_one.setup(dataframe)
    alignn_one.setup()
    alignn_two = Alignn()

    # setup co-Training
    coTraining = CoTraining(alignn_one, alignn_two)
    coTraining.setup_data()
    coTraining.train()
