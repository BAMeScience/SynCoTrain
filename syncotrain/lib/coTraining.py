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
        number_of_classifiers = 2

        y_for_frame = copy.deepcopy(y)
        prediction_results = y_for_frame.to_frame(name='y')
        for col in prediction_results.columns:
            prediction_results[col].values[:] = 0

        y1 = copy.deepcopy(y)
        # y2 = copy.deepcopy(y) # parallel

        test_results = []
        for i in range(int(configuration.config['CoTraining']['steps_of_cotraining'])*number_of_classifiers):
            print("Start CoTraining step: " , int(i/2))
            print("Classifier: " , i%2+1)
            # parallel:
            # results1, test_results1 = self._pu1.train(X, y1)
            # results2, test_results2 = self._pu2.train(X, y2)
            # test_results.append([test_results1, test_results2])
            # y1tmp = copy.deepcopy(y1)
            # y1 = (results2 + y2).clip(0,1)
            # y2 = (results1 + y1tmp).clip(0,1)
            # prediction_results.insert(2*i+0, f"{i}_1", y1, True)
            # prediction_results.insert(2*i+1, f"{i}_2", y2, True)

            # one after each other:
            if i%2==0:
                y1, test_results = self._pu1.train(X, y1)
            else:
                y1, test_results = self._pu2.train(X, y1)
            test_results.append(test_results)
            prediction_results.insert(i, f"{int(i/2)}_{i%2+1}", y1, True)
            # TODO save after each pu-learning?
        return prediction_results.drop(columns=['y']), test_results
