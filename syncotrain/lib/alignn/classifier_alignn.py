import contextlib
import io
import os
import shutil

import pandas as pd
import torch
from alignn.config import TrainingConfig
from alignn.graphs import StructureDataset
from alignn.models.alignn import ALIGNN
from dgl.dataloading import GraphDataLoader
from jarvis.core.atoms import ase_to_atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import loadjson
from alignn.train import train_dgl

from syncotrain.lib import puLearning
from syncotrain.src import configuration
from syncotrain.lib.classifier import Classifier

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


output_dir = None


def get_config(name, train_ratio, val_ratio, test_ratio):
    config = loadjson('syncotrain/lib/alignn/default_class_config.json')
    config["output_dir"] = configuration.result_dir + "/tmp"

    classifier = configuration.config[name]
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
        config = get_config(self._name, 1 - val_ratio, val_ratio, 0)

        train_loader = get_dataloader(config, X, y=y)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                train_dgl(config=config, model=self._model, train_val_test_loaders=[
                    train_loader,  # Training set
                    train_loader,  # Validataion set  TODO split in training and validation set
                    train_loader,  # Test set  TODO set NONE when bug in alignn is fixed
                    train_loader.dataset.prepare_batch])

            # Load best model from file
            self._model = ALIGNN(config.model)
            self._model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pt'))['model'])

        finally:
            if os.path.exists(config.output_dir):
                shutil.rmtree(config.output_dir)

    def predict(self, X: pd.Series):
        config = get_config(self._name, 0, 0, 1)
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
        data = X.to_frame(name='X')
        data.insert(1, 'y', result.detach().numpy(), True)
        return data['y']
