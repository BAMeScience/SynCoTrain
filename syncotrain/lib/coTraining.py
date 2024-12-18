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
        for i in range(int(configuration.config['CoTraining']['steps_of_cotraining'])):
            y = self._pu1.train(X, y)
            y = self._pu2.train(X, y)
        return y
