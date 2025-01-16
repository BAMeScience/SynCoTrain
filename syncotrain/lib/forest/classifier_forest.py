import os
import torch

import pandas as pd
import numpy as np

from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestClassifier
from pymatgen.io.ase import AseAtomsAdaptor

from syncotrain.src import configuration
from syncotrain.lib.classifier import Classifier

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


output_dir = None


class Forest(Classifier):
    def __init__(self, name):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._name = name
        self._model = None

    def fit(self, X: pd.Series, y: pd.Series):
        featurizer = ElementProperty.from_preset("magpie", impute_nan=False)
        X_new = [ featurizer.featurize(AseAtomsAdaptor.get_structure(x).composition) for x in X ]
        self._model = RandomForestClassifier(n_estimators=100)
        self._model.fit(X_new, y)

    def predict(self, X: pd.Series):
        featurizer = ElementProperty.from_preset("magpie", impute_nan=False)
        X_new = [ featurizer.featurize(AseAtomsAdaptor.get_structure(x).composition) for x in X ]
        result = self._model.predict_proba(X_new)
        data = X.to_frame(name='X')
        result_g = torch.from_numpy(np.concatenate(result[0:result.size, 0:1]))
        data.insert(1, 'y', result_g.detach().numpy(), True)
        return data['y']
