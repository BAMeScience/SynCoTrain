import pandas as pd

from src import configuration
from lib.classifier import Classifier
from lib.puLearning import PuLearning


co_step = None
classifier = None


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

    @property
    def pu_classifiers(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self.first_pu_classifier, self.second_pu_classifier

    @property
    def first_pu_classifier(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._pu1

    @first_pu_classifier.setter
    def first_pu_classifier(self, pu1: PuLearning):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._pu1 = pu1

    @property
    def second_pu_classifier(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._pu2

    @second_pu_classifier.setter
    def second_pu_classifier(self, pu2: PuLearning):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._pu2 = pu2

    def setup_data(self):
        pass

    def train(self):
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        # read dataframe from file
        dataframe = get_data()

        global co_step
        global classifier
        print(configuration.config)
        a = configuration.config['General']['steps_of_cotraining']
        for i in range(int(configuration.config['General']['steps_of_cotraining'])):
            co_step = i
            classifier = 1  # 1 for first classifier
            classifier1 = Classifier()

            pulearning1 = PuLearning(self._classifier1)
            pulearning1.setup(dataframe)

            pulearning1.train()
            co_step = i
            classifier = 2  # 2 for second classifier
            pulearning2 = PuLearning(self._classifier2)
            setup_data()
            pulearning2.train()
            # TODO implement
