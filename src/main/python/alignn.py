from typing import List

import pandas as pd
from pandas import DataFrame

from src.main.python import configuration, coTraining, puLearning
from src.main.python.classifier import Classifier
from src.main.resources.crystal_structure_conversion import ase_to_jarvis

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class Alignn(Classifier):
    def setup(self, data: DataFrame, prop: str, TARGET: str):
        # from method from old pu_data_selection file
        # format data for alignn: digits after comma and Nan values and save in new file
        # return f'Data was prepared in {data_files_dir} directory.'
        # the order should be first postive, then unlabeld class

        # format
        data[prop] = data[prop].astype('int16')  # TODO same data? complete?

        # open file and write POSCAR....vasp files
        data_dest = configuration.project_path / configuration.input_dir / 'alignn_format'
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






        # from alignn_pu_config.py
        # TODO think about how to get pu configurations
        max_num_of_iterations = configuration.config['General']['number_of_iterations']
        start_of_iterations = configuration.config['General']['start_iteration']
        data_dir = configuration.input_dir
        root_dir = data_dir + "alignn_format"
        alignn_config_dir = "pu_alignn/alignn_configs"
        default_class_config = alignn_config_dir + "/default_class_config.json"
        class_config_name = alignn_config_dir + f'/class_config_{data_prefix}{experiment}_{prop}.json'
        pu_config_name = alignn_config_dir + f'/pu_config_{data_prefix}{experiment}_{prop}.json'

        pu_setup = dict()
        pu_setup["default_class_config"] = default_class_config
        pu_setup["pu_config_name"] = pu_config_name
        pu_setup["class_config_name"] = class_config_name
        pu_setup["data_dir"] = data_dir
        pu_setup["root_dir"] = root_dir
        pu_setup["file_format"] = "poscar"
        pu_setup["keep_data_order"] = False  # overwrites this attrib in config
        pu_setup["classification_threshold"] = 0.5  # also overwrites if present
        pu_setup["batch_size"] = None
        pu_setup["output_dir"] = None
        pu_setup["epochs"] = 120
        pu_setup["max_num_of_iterations"] = max_num_of_iterations
        pu_setup["start_of_iterations"] = start_of_iterations
        pu_setup["small_data"] = small_data

        print(os.getcwd())
        with open(pu_setup["pu_config_name"], "w+") as configJson:
            json.dump(pu_setup, configJson, indent=2)

        print(f'New PU Alignn pu_config_{data_prefix}{experiment}_{prop}.json was generated.')

        return pu_config_name








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