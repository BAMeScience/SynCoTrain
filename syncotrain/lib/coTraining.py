import copy

import pandas as pd

from syncotrain.src import configuration
from syncotrain.lib.puLearning import PuLearning


class CoTraining:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, pu1: PuLearning, pu2: PuLearning):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._pu1 = pu1
        self._pu2 = pu2

    def train(self, X: pd.Series, y: pd.Series):
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        count_init = y.sum()

        y_for_frame = copy.deepcopy(y)
        prediction_results = y_for_frame.to_frame(name='y')
        for col in prediction_results.columns:
            prediction_results[col].values[:] = 0

        y1 = copy.deepcopy(y)
        y2 = copy.deepcopy(y)
        for i in range(int(configuration.config['CoTraining']['steps_of_cotraining'])):
            print("Start CoTraining step: " , i)
            results1 = self._pu1.train(X, y1)
            results2 = self._pu2.train(X, y2)
            count1 = results1.sum()
            count2 = results2.sum()
            y1tmp = copy.deepcopy(y1)
            y1 = (results2 + y2).clip(0,1)
            y2 = (results1 + y1tmp).clip(0,1)

            prediction_results.insert(2*i+0, f"{i}_1", y1, True)
            prediction_results.insert(2*i+1, f"{i}_2", y2, True)
            # TODO save after each pu-learning
        return prediction_results.drop(columns=['y'])
