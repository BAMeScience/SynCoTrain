import numpy as np
import pandas as pd
from pandas import DataFrame

from src import configuration
from lib.classifier import Classifier


def setup_data(unlabeled_df, positive_df):  # TODO set random_states?
    test_ratio = float(configuration.config['PuLearning']['test_ratio'])

    negative_df = unlabeled_df.sample(n=positive_df['synth'].size, random_state=1)

    negative_test_df = negative_df.sample(frac=test_ratio, random_state=2)
    positive_test_df = positive_df.sample(frac=test_ratio, random_state=3)

    positive_train_df = positive_df.drop(index=positive_test_df.index)
    negative_train_df = negative_df.drop(index=negative_test_df.index)

    test_df = pd.concat([negative_test_df, positive_test_df])
    train_df = pd.concat([negative_train_df, positive_train_df])

    unlabeled_predict_df = unlabeled_df.drop(index=negative_test_df.index).drop(index=negative_train_df.index)

    return test_df, train_df, unlabeled_predict_df[['material_id', 'atoms', 'formation_energy_per_atom', 'energy_above_hull']]


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

    def train(self, data: DataFrame):  # TODO dont use names of columns, give it in config file
        number_of_iterations = int(configuration.config['PuLearning']['number_of_iterations'])  # T

        positive_df = data[data['synth'] == 1]  # P = experimental data
        unlabeled_df = data[data['synth'] == 0]  # U

        leaveout_test_ratio = float(configuration.config['PuLearning']['leaveout_test_ratio'])
        leaveout_df = positive_df.sample(frac=leaveout_test_ratio, random_state=4242)
        positive_df = positive_df.drop(index=leaveout_df.index)

        sum_predict = pd.DataFrame(0, index=np.arange(data['synth'].size), columns=['synth'])
        for i in range(number_of_iterations):
            test_df, train_df, unlabeled_predict_df = setup_data(unlabeled_df, positive_df)
            test_df = pd.concat([test_df, leaveout_df])
            self._classifier.fit(train_df[['material_id', 'atoms']],
                                 train_df[['synth']])
            y_test = self._classifier.predict(test_df[['material_id', 'atoms']])  # TODO how to evaluate test data
            y = self._classifier.predict(unlabeled_predict_df)
            sum_predict.insert(1, f"{i}", y, True)
            sum_predict['synth'] = sum_predict[['synth', f"{i}"]].sum(axis=1)
        sum_predict = sum_predict['synth']
        # next divide with number_of_iterations and set threshold
