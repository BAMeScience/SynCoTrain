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
    def setup(self):
        pass

    @abstractmethod
    def fit(self):
        # input: X data and y labels
        pass # output model?

    @abstractmethod
    def predict(self, data: DataFrame):
        # input X unlabeled Data
        pass # output y predicted labels
