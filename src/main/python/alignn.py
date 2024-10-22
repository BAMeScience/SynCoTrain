from typing import List

from src.main.python.classifier import Classifier

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class Alignn(Classifier):
    def setup(self):
        # from old pu_data_selection file
        from pu_alignn.preparing_data_byFile import prepare_alignn_data
        # ß format data for alignn digits after comma and Nan values and save in new file
        alignn_data_log = prepare_alignn_data(small_data=small_data, experiment=experiment, ehull015=ehull015)
        print(alignn_data_log)

    def fit(self):
        pass

    def predict(self, data: List):
        pass