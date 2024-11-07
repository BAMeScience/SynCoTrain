import pandas as pd
from pandas import DataFrame

from src import configuration
from lib import coTraining
from lib.classifier import Classifier

experimental_datasize = None
split_id_dir_path = None
pu_setup = None


def setup_data(unlabeled_df, positive_df):  # TODO set random_states?
    test_ratio = configuration.config['PuLearning']['test_ratio']

    negative_df = unlabeled_df.sample(n=positive_df['synth'].size, random_state=1)

    negative_test_df = negative_df.sample(frac=test_ratio, random_state=2)
    positive_test_df = positive_df.sample(frac=test_ratio, random_state=3)

    positive_train_df = positive_df.drop(index=positive_test_df.index)
    negative_train_df = negative_df.drop(index=negative_test_df.index)

    test_df = pd.concat([negative_test_df, positive_test_df])
    train_df = pd.concat([negative_train_df, positive_train_df])

    unlabeled_predict_df = unlabeled_df.drop(index=negative_test_df.index).drop(index=negative_train_df.index)

    return test_df, train_df, unlabeled_predict_df


class PuLearning:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, classifier: Classifier):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._classifier = classifier

    def train(self, data: DataFrame):
        number_of_iterations = configuration.config['PuLearning']['number_of_iterations']  # T

        positive_df = data[data['synth'] == 1]  # P = experimental data
        unlabeled_df = data[data['synth'] == 0]  # U

        for i in range(number_of_iterations):
            test_df, train_df, unlabeled_predict_df = setup_data(unlabeled_df, positive_df)
            # TODO call classifier training


# TODO setup and train func
# def functions for etc. data preparation, file save, file read, delete information from df
