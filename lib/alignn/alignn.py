from typing import List

import pandas as pd
from pandas import DataFrame

from src import configuration
from lib import coTraining, puLearning
from lib.classifier import Classifier
from src.utils.crystal_structure_conversion import ase_to_jarvis

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class Alignn(Classifier):
    def __init__(self, config):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._config = config

    @property
    def configuration(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self.configuration

    @configuration.setter
    def configuration(self, config):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._config = config

    def setup(self, data: DataFrame, prop: str, TARGET: str):
        a = self._config
        # from method from old pu_data_selection file
        # format data for alignn: digits after comma and Nan values and save in new file
        # return f'Data was prepared in {data_files_dir} directory.'
        # the order should be first postive, then unlabeld class

        # format
        data[prop] = data[prop].astype('int16')  # TODO same data? complete?

        # open file and write POSCAR....vasp files
        data_dest = configuration.project_path / configuration.input_dir / 'generated/alignn_format'
        data_dest.mkdir(parents=True, exist_ok=True)
        f = open((data_dest / f"{configuration.data_prefix}{prop}_id_from_{TARGET}.csv"), "w")

        # directory of POSCAR....vasp files
        data_files_dir = data_dest / f"{configuration.data_prefix}atomistic_{prop}_co{coTraining.co_step}alignn"  # path configured. old: "...{prop}_{experiment}"
        data_files_dir.mkdir(parents=True, exist_ok=True)

        for _, row in data.iterrows():
            jid = row["material_id"]
            target_value = row[TARGET]
            # The line below keeps NaN values in the dataframe. Used for running small_data experiments.
            formatted_target = "%6f" % target_value if pd.notna(target_value) else "NaN"

            poscar_name = "POSCAR-" + str(jid) + ".vasp"
            jarvisAtom = ase_to_jarvis(row["atoms"])
            jarvisAtom.write_poscar(data_files_dir / poscar_name)
            f.write("%s,%s\n" % (poscar_name, formatted_target))
        f.close()















        # from alignn_pu_learning.py
        puLearning.split_id_dir_path  # old split_id_path

        alignn_dir = "pu_alignn"  # TODO new directory structure
        alignn_config_dir = os.path.join(alignn_dir, "alignn_configs")
        pu_config_name = alignn_pu_config_generator(experiment, small_data, ehull015)
        pu_setup = loadjson(pu_config_name)


    def fit(self):
        pass

    def predict(self, data: List):
        pass