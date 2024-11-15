from abc import ABC, abstractmethod

from pandas import DataFrame


class Classifier(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def setup(self, data: DataFrame):
        # TODO no input needed?
        pass

    @abstractmethod
    def fit(self, X, y):
        # TODO is y a dataframe?
        # input: X data and y labels
        pass

    @abstractmethod
    def predict(self, data):
        # input X unlabeled Data
        # TODO is this needed here?
        y = None  # TODO DataFrame?
        return y
