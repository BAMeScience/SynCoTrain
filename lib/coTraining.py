import pandas as pd

from src import configuration
from lib.classifier import Classifier
from lib.puLearning import PuLearning


co_step = None
classifier = None


def get_data():
    # load dataframe
    df_path = configuration.input_dir + "/" + configuration.config['General']['input_df_file']
    return pd.read_pickle(df_path)


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
            pulearning1 = PuLearning(self._classifier1)
            pulearning1.setup(dataframe)

            pulearning1.train()
            co_step = i
            classifier = 2  # 2 for second classifier
            pulearning2 = PuLearning(self._classifier2)
            pulearning2.setup_data()
            pulearning2.train()
            # TODO implement
