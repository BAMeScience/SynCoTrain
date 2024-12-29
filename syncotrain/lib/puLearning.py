import copy

import pandas as pd

from syncotrain.src import configuration
from syncotrain.lib.classifier import Classifier


def setup_data(random_factor, unlabeled_df, positive_df):  # TODO set random_states?
    test_ratio = float(configuration.config['PuLearning']['test_ratio'])

    # select as much negative data as positive
    negative_df = unlabeled_df.sample(n=positive_df['y'].size, random_state=1+random_factor)

    negative_test_df = negative_df.sample(frac=test_ratio, random_state=2+random_factor)
    positive_test_df = positive_df.sample(frac=test_ratio, random_state=3+random_factor)

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

        # combine X and y into a data frame
        data = X.to_frame(name='X')
        data.insert(1, 'y', y, True)

        # split in positive and negative (unlabeled) data
        positive_df = data[data['y'] == 1]  # P = experimental data
        unlabeled_df = data[data['y'] == 0]  # U

        # select some leaveout data and separate them from the positive data
        leaveout_test_ratio = float(configuration.config['PuLearning']['leaveout_test_ratio'])
        leaveout_df = positive_df.sample(frac=leaveout_test_ratio, random_state=4242)
        positive_df = positive_df.drop(index=leaveout_df.index)

        # initialise prediction_score with ids and initially set all to zero
        y_for_frame = copy.deepcopy(y)
        prediction_score = y_for_frame.to_frame(name='y')
        for col in prediction_score.columns:
            prediction_score[col].values[:] = 0

        # start iterations for pu-learning
        for i in range(number_of_iterations):
            print("Start Iteration: " , i)

            # make a new copy of classifier for each iteration
            new_classifier = copy.deepcopy(self._classifier)

            # split data in test, train data and data that needs to be predicted
            test_df, train_df, unlabeled_predict_df = setup_data(i, unlabeled_df, positive_df)

            # add leaveout data to test data
            test_df = pd.concat([test_df, leaveout_df])

            # train classifier with train data
            new_classifier.fit(train_df['X'], train_df['y'])

            # test performance with test data
            # y_test = new_classifier.predict(test_df['X'])
            # test_results = test_performance(y_test)  # TODO how to evaluate test data and what to do with leaveout data

            # get predictions for unlabeled data
            prediction = new_classifier.predict(unlabeled_predict_df['X'])

            # use predictions to add up a score
            prediction_score.insert(1, f"{i}", prediction, True)
            prediction_score['y'] = prediction_score[['y', f"{i}"]].sum(axis=1)

        # divide by number_of_iterations and decide based on the threshold
        prediction_score = prediction_score['y'].div(number_of_iterations).round(2)
        results = (prediction_score > float(configuration.config['PuLearning']['prediction_threshold'])).astype(int)

        return results
