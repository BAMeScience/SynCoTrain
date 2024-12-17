from __future__ import annotations

import argparse
import os

import pandas as pd

from syncotrain.lib.puLearning import PuLearning
from syncotrain.src import configuration
from syncotrain.lib.alignn.classifier_alignn import Alignn
from syncotrain.lib.coTraining import CoTraining
from syncotrain.src.configuration import configure


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
    df_path = configuration.input_dir + "/new/" + configuration.config['General']['input_df_file']
    #df_path = configuration.input_dir + "/" + configuration.config['General']['input_df_file']
    return pd.read_pickle(df_path)

def get_smaller_data():
    df_path = configuration.input_dir + "/" + configuration.config['General']['input_df_file']
    df = pd.read_pickle(df_path)
    df_new = df[['material_id', 'atoms', 'formation_energy_per_atom', 'energy_above_hull', 'synth']].sample(frac=0.1, random_state=42)
    df_new.to_pickle(configuration.input_dir + "/new/" + configuration.config['General']['input_df_file'])
    return df_new


if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    # get user input
    ehull015, small_data = get_user_input()
    # read config file and set configurations
    configure(ehull015, small_data)

    # read dataframe
    df = get_data()
    #df = get_smaller_data()

    # setup classifers
    alignn_one = Alignn('Classifier1')
    alignn_two = Alignn('Classifier2')

    # setup pu_learning for them
    puLearning_one = PuLearning(alignn_one)
    puLearning_two = PuLearning(alignn_two)

    # setup co_training
    co_training = CoTraining(puLearning_one, puLearning_two)

    # start training
    results = co_training.train(df['atoms'],df['synth'])
