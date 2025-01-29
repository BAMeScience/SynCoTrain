from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

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
    #df_path = configuration.input_dir + "/new/" + configuration.config['General']['input_df_file']
    df_path = configuration.input_dir + "/" + configuration.config['General']['input_df_file']
    return pd.read_pickle(df_path)

def get_smaller_data():
    df_path = configuration.input_dir + "/" + configuration.config['General']['input_df_file']
    df = pd.read_pickle(df_path)
    df_new = df[['material_id', 'atoms', 'formation_energy_per_atom', 'energy_above_hull', 'synth']].sample(frac=0.2, random_state=42)
    df_new.to_pickle(configuration.input_dir + "/new/" + configuration.config['General']['input_df_file'])
    return df_new


def evaluate(results: pd.DataFrame, gt: pd.Series):
    co_steps = int(configuration.config['CoTraining']['steps_of_cotraining'])
    results.insert(co_steps, 'gt', gt, True)
    co_list = []
    for co in range(co_steps):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index, row in results.iterrows():
            if row['gt'] == row[co]:
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
        positive = (tp+fp)/(tp+fp+tn+fn)
        co_list.append([round(precision, 2), round(recall, 2), round(f1, 2), round(positive, 2)])
    return co_list
                    

def plot(value_list: list):
    (configuration.project_path / configuration.result_dir / "plots").mkdir(parents=True, exist_ok=True)
    precision = []
    recall = []
    f1 = []
    positiv_rate = []
    for i in value_list:
        precision.append(i[0])
        recall.append(i[1])
        f1.append(i[2])
        positiv_rate.append(i[3])
    x_all = ["Alignn0", "RandomForest1", "Alignn2", "RandomForest3", "Alignn4", "RandomForest5"]
    x = x_all[:len(positiv_rate)]
    plt.figure(0)
    plt.title(f"Recall/Positive rate")
    plt.xlabel('Co-Training step')
    plt.plot(x, recall, 'r-', label='Recall')
    plt.plot(x, positiv_rate, 'b-', label='Predicted positive rate')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(f"data/results/plots/recall.png")
    plt.title(f"Scores")
    plt.xlabel('Co-Training step')
    plt.plot(x, precision, 'g-', label='Precision')
    plt.plot(x, f1, 'y-', label='F1-score')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(f"data/results/plots/scores.png")


def plot_test(value_list):
    (configuration.project_path / configuration.result_dir / "plots").mkdir(parents=True, exist_ok=True)
    step = 0
    for co_step in value_list:
        test = []
        leaveout = []
        if co_step:
            for pu_iteration in co_step:
                test.append(pu_iteration[0])
                leaveout.append((pu_iteration[1]))
            x = [k for k in range(len(test))]
            plt.figure(1+step)
            plt.title(f"Co-Training step {step}")
            ax = plt.gca()
            ax.set_xticks(x)
            plt.xlabel('Pu-Learning iteration')
            plt.plot(x, test, 'r-', label='test data')
            plt.plot(x, leaveout, 'b-', label='leaveout data')
            plt.legend(loc='lower right', frameon=True)
            plt.savefig(f"data/results/plots/test_co{step}.png")
        step += 1 


if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    # get user input
    ehull015, small_data = get_user_input()
    # read config file and set configurations
    configure(ehull015, small_data)

    # read dataframe and select ground truth
    df = get_data()
    #df = get_smaller_data()
    gt = (df['energy_above_hull'] <= 0.015).astype(int)

    # setup classifers
    alignn_one = Alignn('Classifier1')
    forest_one = Forest('Classifier2')

    # setup pu_learning for them
    puLearning_one = PuLearning(alignn_one, gt)
    puLearning_two = PuLearning(forest_one, gt)

    # setup co_training
    co_training = CoTraining(puLearning_one, puLearning_two)

    # start training
    results, test_results = co_training.train(df['atoms'],df['synth'])  # stability nicht

    # evaluate and plot results
    calc_values = evaluate(results, gt)
    plot(calc_values)
    if test_results:
        plot_test(test_results)
