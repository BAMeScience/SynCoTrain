import contextlib
import csv
import io
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from alignn.config import TrainingConfig
from alignn.graphs import StructureDataset
from alignn.models.alignn import ALIGNN
from alignn.models.alignn_atomwise import ALIGNNAtomWiseConfig, ALIGNNAtomWise
from dgl.dataloading import GraphDataLoader
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import loadjson
from pandas import DataFrame
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl

from syncotrain.lib import puLearning
from syncotrain.src import configuration
from syncotrain.lib.classifier import Classifier
from syncotrain.src.utils.crystal_structure_conversion import ase_to_jarvis  # TODO whats wrong here?

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


output_dir = None
# TODO replace 'synth' here and everywhere


def get_config(i, train_ratio, val_ratio, test_ratio):  # TODO how to call Classifier1,...?
    config = loadjson('syncotrain/lib/alignn/default_class_config.json')  # TODO
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
    config["write_checkpoint"] = True
    (configuration.project_path / config["output_dir"]).mkdir(parents=True, exist_ok=True)
    global output_dir
    output_dir = config["output_dir"]
    config = TrainingConfig(**config)
    return config


def get_dataloader(config, X: pd.Series, y: pd.Series = None, drop_last=True, shuffle=True):
    dataset = get_torch_dataset(
        X, y,
        neighbor_strategy=config.neighbor_strategy,
        # TODO: Add more config options here...
        classification=config.model.classification,
    )

    loader = GraphDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_line_graph,
        drop_last=drop_last,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_ddp=False,
    )
    return loader


def load_graphs(
    df               : pd.DataFrame,
    cutoff           : float = 8,
    max_neighbors    : int   = 12,
    use_canonize     : bool  = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        atoms = ase_to_atoms(atoms)
        return Graph.atom_dgl_multigraph(
            atoms,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
        )

    graphs = df["X"].apply(atoms_to_graph).values

    return graphs


def get_torch_dataset(
    X                : pd.Series,
    y                : pd.Series = None,
    id_tag           : str   = "jid",
    target           : str   = "y",
    neighbor_strategy: str   = "k-nearest",
    atom_features    : str   = "cgcnn",
    use_canonize     : bool  = False,
    line_graph       : bool  = True,
    cutoff           : float = 8.0,
    max_neighbors    : int   = 12,
    classification   : bool  = False,
):
    """Get Torch Dataset."""

    # In case we only want to make predictions
    if y is None:
        y = float('nan')

    bla = X
    bli = y
    df = pd.DataFrame({'X': X, 'y': y, id_tag: range(len(X))})

    # Suppress annoying output to stdout
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # Convert list of atoms to list of graphs
        graphs = load_graphs(
            df,
            use_canonize      = use_canonize,
            cutoff            = cutoff,
            max_neighbors     = max_neighbors,
        )
        # Create data set for training/predicting
        data = StructureDataset(
            df,
            graphs,
            target            = target,
            atom_features     = atom_features,
            line_graph        = line_graph,
            id_tag            = id_tag,
            classification    = classification,
        )
    return data


class Alignn(Classifier):
    def __init__(self, name):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._name = name
        self._model = None

    def fit(self, X: pd.Series, y: pd.Series):
        val_ratio = float(configuration.config['PuLearning']['validation_ratio']) / (1 - float(
            configuration.config['PuLearning']['test_ratio']))
        config = get_config(1, 1 - val_ratio, val_ratio, 0)

        train_loader = get_dataloader(config, X, y=y)
        #test_dummy = get_dataloader(config, X)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                train_dgl(config=config, model=self._model, train_val_test_loaders=[
                    train_loader,  # Training set
                    train_loader,  # Validataion set (TODO)   # TODO how to handle non natural numbers?
                    train_loader,  # Test set # TODO set NONE when bug in alignn is fixed
                    train_loader.dataset.prepare_batch])

            # Load best model from file
            self._model = ALIGNN(config.model)
            self._model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pt'))['model'])

        finally:
            if os.path.exists(config.output_dir):
                # shutil.rmtree(config.output_dir)
                pass

    def predict(self, X: pd.Series):
        config = get_config(1, 0, 0, 1)
        predict_loader = get_dataloader(config, X, drop_last=False, shuffle=False)

        # Set model to evaluation
        self._model.eval()

        result = torch.tensor([])

        for batch in predict_loader:
            result = torch.concat(
                (result, self._model(batch[0:2])),
                axis=0
            )

        # `result` contains a (n, 2)-matrix, with log-probabilities
        # for the two classes. Return probability for class `1` as
        # pandas Series
        result = result[:, 1].exp()
        result = pd.Series(result.detach().numpy())
        return result







#     def fit_old(self, X: pd.Series, y: pd.Series):  # TODO where is csv used? Do I need it without script call? Should I better call by script?
#         val_ratio = float(configuration.config['PuLearning']['validation_ratio'])/(1-float(configuration.config['PuLearning']['test_ratio']))  # TODO how to handle non natural numbers?
#         config = get_config(1, 1-val_ratio-0.05, val_ratio-0.05, 0.1)
#
#         X.insert(2, 'synth', y, True)  # note that X contains y now
#         csv_file = update_csv_file(X)
#         dataset = get_dataset(csv_file)
#         self.train_alignn(config, dataset)
#
#     def predict_old(self, X):
#         #config = get_config(1, 0.1, 0.1, 0.8)
#         X.insert(2, 'synth', 0, True)
#         csv_file = update_csv_file(X)
#         dataset = get_dataset(csv_file)
#         return self.test_alignn(dataset)
#         #return load_results()
#
#     def train_alignn(self, config, dataset):
#         (train_loader, val_loader, test_loader, prepare_batch) = get_train_val_loaders(
#             dataset_array=dataset,
#             target=config.target,
#             train_ratio=config.train_ratio,
#             val_ratio=config.val_ratio,
#             test_ratio=config.test_ratio,
#             batch_size=config.batch_size,
#             atom_features=config.atom_features,
#             split_seed=config.random_seed,
#             neighbor_strategy=config.neighbor_strategy,
#             standardize=config.atom_features != "cgcnn",
#             id_tag=config.id_tag,
#             pin_memory=config.pin_memory,
#             workers=config.num_workers,
#             save_dataloader=config.save_dataloader,
#             use_canonize=config.use_canonize,
#             filename=config.filename,
#             cutoff=config.cutoff,
#             max_neighbors=config.max_neighbors,
#             output_features=config.model.output_features,
#             classification_threshold=config.classification_threshold,
#             target_multiplication_factor=config.target_multiplication_factor,
#             standard_scalar_and_pca=config.standard_scalar_and_pca,
#             keep_data_order=config.keep_data_order,
#             output_dir=config.output_dir,
#         )
#         train_dgl(config, model=self._model,
#                   train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch])
#
#         # Load best model from file
#         self._model = ALIGNN(config.model)
#         model_path = output_dir + '/best_model.pt'
#         var = self._model.load_state_dict(torch.load(model_path)['model'])
#         x = 0
#
#     def test_alignn(self, dataset):
#         device = "cpu"
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         model = ALIGNN()
#         model.load_state_dict(torch.load("checkpoint_250.pt")["model"])
#         model.to(device)
#         model.eval()
#         atoms = Atoms.from_poscar("POSCAR")
#
#         g, lg = Graph.atom_dgl_multigraph(atoms)
#         out_data = (
#             model([g.to(device), lg.to(device)])
#             .detach()
#             .cpu()
#             .numpy()
#             .flatten()
#             .tolist()[0]
#         )
#         print("original", out_data)
#         return out_data
#
#     def setup(self, data):
#         # make initial csv_file and poscar-files for alignn call
#         prepare_data(data)
#
#
# def prepare_data(data: DataFrame):
#     # if paths dont exist, make directories
#     directory = configuration.project_path / configuration.input_dir / "generated/alignn_format"
#     directory.mkdir(parents=True, exist_ok=True)
#     poscar_dir = directory / "poscar"
#     poscar_dir.mkdir(parents=True, exist_ok=True)
#
#     # write mapping in csv-file and create poscar files
#     csv_file = directory / "id_prop.csv"
#     file = open(csv_file, "w")  # TODO put this in loop and write if first time ...
#     for _, row in data.iterrows():
#         # Save atomic structure to POSCAR file
#         poscar_name = f"POSCAR-{row['material_id']}.vasp"
#         jarvis_atom = ase_to_jarvis(row["atoms"])
#         jarvis_atom.write_poscar(poscar_dir / poscar_name)
#
#         # Write the mapping of POSCAR files to target values
#         target_value = row['synth']  # target = positive/negative
#         formatted_target = f"{target_value:.6f}" if pd.notna(target_value) else "NaN"
#         file.write(f"{poscar_name},{formatted_target}\n")
#     file.close()
#     return csv_file
#
# def update_csv_file(data):
#     directory = configuration.project_path / configuration.input_dir / "generated/alignn_format"
#     csv_file = directory / "id_prop.csv"
#     file = open(csv_file, "w")
#     for _, row in data.iterrows():
#         target_value = row['synth']  # target = positive/negative
#         formatted_target = f"{target_value:.6f}" if pd.notna(target_value) else "NaN"
#         file.write(f"POSCAR-{row['material_id']}.vasp,{formatted_target}\n")
#     file.close()
#     return csv_file
#
# def load_results():
#     output_file = "prediction_results_test_set.csv"
#     path = output_dir / output_file
#     results = pd.read_csv(path)
#     return results[['prediction']]
#
# def get_dataset(csv_file):
#     with open(csv_file, "r") as f:
#         reader = csv.reader(f)
#         data = [row for row in reader]
#     dataset = []
#     for i in data:
#         info = {}
#         file_name = i[0]
#         atoms = Atoms.from_poscar(configuration.project_path / configuration.input_dir / "generated/alignn_format/poscar" / file_name)
#
#         info["atoms"] = atoms.to_dict()
#         info["jid"] = file_name
#
#         tmp = [float(j) for j in i[1:]]  # float(i[1])
#         if len(tmp) == 1:
#             tmp = tmp[0]
#         info["target"] = tmp  # float(i[1])
#         dataset.append(info)
#     return dataset