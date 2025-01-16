from __future__ import annotations

import argparse
import os

import pandas as pd

from syncotrain.lib.puLearning import PuLearning
from syncotrain.src import configuration
from syncotrain.lib.alignn.classifier_alignn import Alignn
from syncotrain.lib.forest.classifier_forest import Forest
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
    df_new = df[['material_id', 'atoms', 'formation_energy_per_atom', 'energy_above_hull', 'synth']].sample(frac=0.3, random_state=42)
    df_new.to_pickle(configuration.input_dir + "/new/" + configuration.config['General']['input_df_file'])
    return df_new


def evaluate(results: pd.DataFrame, gt: pd.Series):
    co_steps = int(configuration.config['CoTraining']['steps_of_cotraining'])
    #pu_iterations = int(configuration.config['PuLearning']['number_of_iterations'])
    #shape_rows, shape_columns = df.shape
    results.insert(co_steps*2, 'gt', gt, True)
    co_list = []
    for co in range(co_steps):
        i_list = []
        for i in range(2):
            row_name = f"{co}_{i+1}"
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for index, row in results.iterrows():
                if row['gt'] == row[row_name]:
                    if row['gt'] == 0:
                        tn += 1
                    else:
                        tp += 1
                else:
                    if row['gt'] == 0:
                        fp += 1
                    else:
                        fn += 1
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*precision*recall/(precision+recall)
            i_list.append([round(precision, 2), round(recall, 2), round(f1, 2)])
        co_list.append(i_list)
    return co_list
                    



if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    # get user input
    ehull015, small_data = get_user_input()
    # read config file and set configurations
    configure(ehull015, small_data)

    # read dataframe and select ground truth
    #df = get_data()
    df = get_smaller_data()
    gt = (df['energy_above_hull'] <= 0.015).astype(int)  # TODO how to give it to pu-learning?

    # setup classifers
    alignn_one = Alignn('Classifier1')
    #alignn_two = Alignn('Classifier2')
    forest_one = Forest('Classifier2')

    # setup pu_learning for them
    puLearning_one = PuLearning(alignn_one, gt)
    #puLearning_two = PuLearning(alignn_two)
    puLearning_two = PuLearning(forest_one)

    # setup co_training
    co_training = CoTraining(puLearning_one, puLearning_two)

    # start training
    results, test_results = co_training.train(df['atoms'],df['synth'])
    tptnfpfn = evaluate(results, gt)
    x = 1
