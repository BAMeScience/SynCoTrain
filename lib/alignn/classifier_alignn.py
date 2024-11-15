import pandas as pd
#from alignn import train_alignn
from jarvis.db.jsonutils import loadjson, dumpjson
from pandas import DataFrame

from src import configuration
from lib.classifier import Classifier
from src.utils.crystal_structure_conversion import ase_to_jarvis  # TODO whats wrong here?
#from alignn.data import get_train_val_loaders
#from alignn.train import train_dgl

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


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


def update_csv_file(data):
    directory = configuration.project_path / configuration.input_dir / "generated/alignn_format"
    csv_file = directory / "id_prop.csv"
    file = open(csv_file, "w")
    for _, row in data.iterrows():
        target_value = row['synth']  # target = positive/negative
        formatted_target = f"{target_value:.6f}" if pd.notna(target_value) else "NaN"
        file.write(f"POSCAR-{row['material_id']}.vasp,{formatted_target}\n")
    file.close()


def get_config(i, n_train, n_val, n_test):  # TODO how to call Classifier1,...?
    config = loadjson('lib/alignn/default_class_config.json')
    config['n_train'] = n_train
    config['n_val'] = n_val
    config['n_test'] = n_test

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
    # dumpjson(data=config, filename=f'tmp_config{i}.json')  # TODO change filename?
    return config


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

    def fit(self, X, y):  # TODO where is csv used? Do I need it without script call? Should I better call by script?
        n_val = y.size * float(configuration.config['PuLearning']['validation_ratio'])
        n_test = y.size * float(configuration.config['PuLearning']['test_ratio'])  # TODO how to handle non natural numbers?
        n_train = y.size - n_val - n_test

        config = get_config(1, n_train, n_val, n_test)
        X.insert(2, 'synth', y, True)
        prepare_data(X)  # TODO note that X contains now y
        #!train_alignn.py --root_dir "Out" --config "tmp_config1.json" --output_dir="temp"
        # (train_loader, val_loader, test_loader, prepare_batch) = get_train_val_loaders(
        #     dataset_array=dataset,  # TODO check this
        #     target=config.target,
        #     val_ratio=config.val_ratio,
        #     batch_size=config.batch_size,
        #     atom_features=config.atom_features,
        #     split_seed=config.random_seed,
        #     neighbor_strategy=config.neighbor_strategy,
        #     standardize=config.atom_features != "cgcnn",
        #     id_tag=config.id_tag,
        #     pin_memory=config.pin_memory,
        #     workers=config.num_workers,
        #     save_dataloader=config.save_dataloader,
        #     use_canonize=config.use_canonize,
        #     filename=config.filename,
        #     cutoff=config.cutoff,
        #     max_neighbors=config.max_neighbors,
        #     output_features=config.model.output_features,
        #     classification_threshold=config.classification_threshold,
        #     target_multiplication_factor=config.target_multiplication_factor,
        #     standard_scalar_and_pca=config.standard_scalar_and_pca,
        #     keep_data_order=config.keep_data_order,
        #     output_dir=config.output_dir,
        # )
        # train_dgl(config, model=self._model,
        #           train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch])

    def predict(self, data: DataFrame):
        y = None  # TODO DataFrame?
        return y
