from abc import ABC, abstractmethod

import pandas as pd


class Classifier(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def fit(self, X: pd.Series, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.Series):
        return
