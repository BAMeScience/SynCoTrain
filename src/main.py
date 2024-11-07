from __future__ import annotations

import argparse
import os
import pandas as pd
from pandas import DataFrame

# from lib.puLearning import PuLearning
from src import configuration
from lib.alignn.alignn import Alignn
from lib.coTraining import CoTraining, get_data
from src.configuration import configure


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


def setup_data(data: DataFrame):
    number_of_iterations = 60  # T
    test_ratio = 0.2

    positive_df = data[data['synth'] == 1]  # P = experimental data
    unlabeled_df = data[data['synth'] == 0]  # U
    positive_size = positive_df['synth'].size  # K

    for i in range(number_of_iterations):
        negative_df = unlabeled_df.sample(n=positive_size, random_state=i)

        negative_test_df = negative_df.sample(frac=test_ratio, random_state=i+1)
        positive_test_df = positive_df.sample(frac=test_ratio, random_state=i+2)

        positive_train_df = positive_df.drop(index=positive_test_df.index)
        negative_train_df = negative_df.drop(index=negative_test_df.index)

        test_df = pd.concat([negative_test_df, positive_test_df])
        train_df = pd.concat([negative_train_df, positive_train_df])

        unlabeled_predict_df = unlabeled_df.drop(index=negative_test_df.index).drop(index=negative_train_df.index)
    x = 3
    pass


def get_data():
    # load dataframe
    df_path = configuration.input_dir + "/new/" + configuration.config['General']['input_df_file']
    return pd.read_pickle(df_path)


if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    # get user input
    ehull015, small_data = get_user_input()
    # read config file and set configurations
    configure(ehull015, small_data)
    # more setup needed?

    # read dataframe
    df = get_data()

    ##########TEST##############
    # df_path = configuration.input_dir + "/new/" + configuration.config['General']['input_df_file']
    # df = pd.read_pickle(df_path)
    # df_new = df[['material_id', 'atoms', 'formation_energy_per_atom', 'energy_above_hull', 'synth']]#.head(100)
    # df_new.to_pickle(configuration.input_dir + "/new/" + configuration.config['General']['input_df_file'])
    # setup_data(df)
    ############################

    # setup classifers and pu_learning for them
    # alignn_one = Alignn(configuration.config['Classifier1'])
    # puLearning_one = PuLearning(alignn_one)
    # puLearning_one.setup(dataframe)
    # alignn_one.setup()
    # alignn_two = Alignn()

    # setup co-Training
    # coTraining = CoTraining(PuLearning(alignn_one), PuLearning(alignn_two))
    # coTraining.setup_data()
    # coTraining.train()
