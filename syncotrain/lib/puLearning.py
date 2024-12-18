import pandas as pd

from syncotrain.src import configuration
from syncotrain.lib.classifier import Classifier


def setup_data(unlabeled_df, positive_df):  # TODO set random_states?
    test_ratio = float(configuration.config['PuLearning']['test_ratio'])

    negative_df = unlabeled_df.sample(n=positive_df['y'].size, random_state=1)

    negative_test_df = negative_df.sample(frac=test_ratio, random_state=2)
    positive_test_df = positive_df.sample(frac=test_ratio, random_state=3)

    positive_train_df = positive_df.drop(index=positive_test_df.index)
    negative_train_df = negative_df.drop(index=negative_test_df.index)

    test_df = pd.concat([negative_test_df, positive_test_df])
    train_df = pd.concat([negative_train_df, positive_train_df])

    unlabeled_predict_df = unlabeled_df.drop(index=negative_test_df.index).drop(index=negative_train_df.index)

    return test_df, train_df, unlabeled_predict_df


class PuLearning:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, classifier: Classifier):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._classifier = classifier

    def train(self, X: pd.Series, y: pd.Series):
        number_of_iterations = int(configuration.config['PuLearning']['number_of_iterations'])  # T

        data = X.to_frame(name='X')
        data.insert(1, 'y', y, True)
        positive_df = data[data['y'] == 1]  # P = experimental data
        unlabeled_df = data[data['y'] == 0]  # U

        leaveout_test_ratio = float(configuration.config['PuLearning']['leaveout_test_ratio'])
        leaveout_df = positive_df.sample(frac=leaveout_test_ratio, random_state=4242)
        positive_df = positive_df.drop(index=leaveout_df.index)

        sum_predict = y.to_frame(name='y')
        for col in sum_predict.columns:
            sum_predict[col].values[:] = 0
        for i in range(number_of_iterations):
            print("Start Iteration: " , i)
            test_df, train_df, unlabeled_predict_df = setup_data(unlabeled_df, positive_df)
            test_df = pd.concat([test_df, leaveout_df])
            self._classifier.fit(train_df['X'], train_df['y'])  # TODO call new instance of alignn every time
            # y_test = self._classifier.predict(test_df['X'])  # TODO how to evaluate test data
            prediction = self._classifier.predict(unlabeled_predict_df['X'])
            sum_predict.insert(1, f"{i}", prediction, True)
            sum_predict['y'] = sum_predict[['y', f"{i}"]].sum(axis=1)
        # divide with number_of_iterations and set threshold
        sum_predict = sum_predict['y'].div(number_of_iterations).round(2)
        results = (sum_predict > float(configuration.config['PuLearning']['prediction_threshold'])).astype(int)
        return results
