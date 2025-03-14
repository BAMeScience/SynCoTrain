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
        sizey = y1.sum()

        test_results = []
        #leave_results = []
        for i in range(int(configuration.config['CoTraining']['steps_of_cotraining'])):
            print("Start CoTraining step: " , i)
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
                results1, test1, leave = self._pu1.train(X, y1)
            else:
                results1, test1, leave = self._pu2.train(X, y1)  # RandomForest liefer 0 positive
            test_results.append(test1)
            #leave_results.append(leave)
            size = results1.sum()
            y1 = (results1 + y1).clip(0,1)
            sizey1 = y1.sum()
            prediction_results.insert(i, f"{i}", y1, True)
            if i == 0:
                leave_results = leave.to_frame(name='0')
            else:
                leave_results.insert(i, f"{i}", leave, True)
            # TODO save after each pu-learning?
        return prediction_results.drop(columns=['y']), test_results, leave_results
