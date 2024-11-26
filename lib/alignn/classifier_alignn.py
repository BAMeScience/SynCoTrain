import csv
import os

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson
from pandas import DataFrame

from lib import puLearning
from src import configuration
from lib.classifier import Classifier
from src.utils.crystal_structure_conversion import ase_to_jarvis  # TODO whats wrong here?
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


output_dir = None
# TODO replace 'synth' here and everywhere


def prepare_data(data: DataFrame):
    # if paths dont exist, make directories
    directory = configuration.project_path / configuration.input_dir / "generated/alignn_format"
    directory.mkdir(parents=True, exist_ok=True)
    poscar_dir = directory / "poscar"
    poscar_dir.mkdir(parents=True, exist_ok=True)

    # write mapping in csv-file and create poscar files
    csv_file = directory / "id_prop.csv"
    file = open(csv_file, "w")  # TODO put this in loop and write if first time ...
    for _, row in data.iterrows():
        # Save atomic structure to POSCAR file
        poscar_name = f"POSCAR-{row['material_id']}.vasp"
        jarvis_atom = ase_to_jarvis(row["atoms"])
        jarvis_atom.write_poscar(poscar_dir / poscar_name)

        # Write the mapping of POSCAR files to target values
        target_value = row['synth']  # target = positive/negative
        formatted_target = f"{target_value:.6f}" if pd.notna(target_value) else "NaN"
        file.write(f"{poscar_name},{formatted_target}\n")
    file.close()
    return csv_file


def update_csv_file(data):
    directory = configuration.project_path / configuration.input_dir / "generated/alignn_format"
    csv_file = directory / "id_prop.csv"
    file = open(csv_file, "w")
    for _, row in data.iterrows():
        target_value = row['synth']  # target = positive/negative
        formatted_target = f"{target_value:.6f}" if pd.notna(target_value) else "NaN"
        file.write(f"POSCAR-{row['material_id']}.vasp,{formatted_target}\n")
    file.close()
    return csv_file


def get_config(i, train_ratio, val_ratio, test_ratio):  # TODO how to call Classifier1,...?
    config = loadjson('lib/alignn/default_class_config.json')
    config["output_dir"] = configuration.result_dir + "/tmp/" + configuration.config['General']['input_df_file'] + f"/classifier_{i}/{puLearning.iteration}"

    classifier = configuration.config[f'Classifier{i}']
    for key in classifier:
        typ = type(config[f"{key}"]).__name__
        if typ == "int":
            value = int(classifier[f"{key}"])
        elif typ == "float":
            value = float(classifier[f"{key}"])
        elif typ == "bool":
            value = bool(classifier[f"{key}"])
        else:
            value = classifier[f"{key}"]
        config[f"{key}"] = value

    config["train_ratio"] = train_ratio
    config["val_ratio"] = val_ratio
    config["test_ratio"] = test_ratio
    (configuration.project_path / config["output_dir"]).mkdir(parents=True, exist_ok=True)
    global output_dir
    output_dir = config["output_dir"]
    return config


def get_dataset(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    dataset = []
    for i in data:
        info = {}
        file_name = i[0]
        atoms = Atoms.from_poscar(configuration.project_path / configuration.input_dir / "generated/alignn_format/poscar" / file_name)

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        info["target"] = tmp  # float(i[1])
        dataset.append(info)
    return dataset


def load_results():
    output_file = "prediction_results_test_set.csv"
    path = output_dir / output_file
    results = pd.read_csv(path)
    return results[['prediction']]


class Alignn(Classifier):
    def __init__(self, name):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._name = name
        self._model = None

    def setup(self, data):
        # make initial csv_file and poscar-files for alignn call
        prepare_data(data)

    def call_alignn(self, config, dataset):
        (train_loader, val_loader, test_loader, prepare_batch) = get_train_val_loaders(
            dataset_array=dataset,
            target=config["target"],
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"],
            test_ratio=config["test_ratio"],
            batch_size=config["batch_size"],
            atom_features=config["atom_features"],
            split_seed=config["random_seed"],
            neighbor_strategy=config["neighbor_strategy"],
            standardize=config["atom_features"] != "cgcnn",
            id_tag=config["id_tag"],
            pin_memory=config["pin_memory"],
            workers=config["num_workers"],
            save_dataloader=config["save_dataloader"],
            use_canonize=config["use_canonize"],
            filename=config["filename"],
            cutoff=config["cutoff"],
            max_neighbors=config["max_neighbors"],
            output_features=config["model"]["output_features"],
            classification_threshold=config["classification_threshold"],
            target_multiplication_factor=config["target_multiplication_factor"],
            standard_scalar_and_pca=config["standard_scalar_and_pca"],
            keep_data_order=config["keep_data_order"],
            output_dir=config["output_dir"],
        )
        train_dgl(config, model=self._model,
                  train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch])

    def fit(self, X, y):  # TODO where is csv used? Do I need it without script call? Should I better call by script?
        val_ratio = float(configuration.config['PuLearning']['validation_ratio'])/(1-float(configuration.config['PuLearning']['test_ratio']))  # TODO how to handle non natural numbers?
        config = get_config(1, 1-val_ratio, val_ratio, 0)

        X.insert(2, 'synth', y, True)  # note that X contains y now
        csv_file = update_csv_file(X)
        dataset = get_dataset(csv_file)
        self.call_alignn(config, dataset)

    def predict(self, X):
        config = get_config(1, 0.8, 0.1, 0.1)
        X.insert(2, 'synth', 0, True)
        csv_file = update_csv_file(X)
        dataset = get_dataset(csv_file)
        self.call_alignn(config, dataset)
        return load_results()
