import pandas as pd

from src.main.python import configuration
from src.main.python.classifier import Classifier
from src.main.python.puLearning import PuLearning


class CoTraining:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, classifier1: Classifier, classifier2: Classifier):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._classifier1 = classifier1
        self._classifier2 = classifier2

    @property
    def classifiers(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self.first_classifier, self.second_classifier

    @property
    def first_classifier(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._classifier1

    @first_classifier.setter
    def first_classifier(self, classifier1: Classifier):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._classifier1 = classifier1

    @property
    def second_classifier(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._classifier2

    @second_classifier.setter
    def second_classifier(self, classifier2: Classifier):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._classifier2 = classifier2

    def setup_data(self):
        pass

    def train(self, classifier1: Classifier, classifier2: Classifier):
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        for i in range(configuration.config['General']['steps_of_coTraining']):
            pulearning1 = PuLearning(classifier1)
            pulearning1.setup_data()
            pulearning1.train()
            pulearning2 = PuLearning(classifier2)
            pulearning2.setup_data()
            pulearning2.train()
            # TODO implement
