from pathlib import Path
from typing import Literal
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd


def list_test_predictions(category: Literal["ppi", "pli"] | None = None) -> list[str]:
    """
    List all system IDs for test systems with predictions.

    Parameters
    ----------
    category : {'ppi', 'pli'}, optional
        If set only return systems in the given category.

    Returns
    -------
    system_ids : list of str
        List of test system IDs.
    """
    system_ids = sorted(
        [
            p.name
            for p in (Path(__file__).parent / "data" / "predictions").iterdir()
            if p.is_dir()
        ]
    )
    if category is not None:
        metrics_table = pd.read_csv(Path(__file__).parent / "data" / "ref_metrics.csv")
        metrics_table = metrics_table[metrics_table["category"] == category]
        system_ids = [
            id for id in system_ids if id in metrics_table["system_id"].to_list()
        ]
    return system_ids


def list_test_pdb_files() -> list[Path]:
    """
    List all structure files for all test systems taken from the PDB.

    Returns
    -------
    files : list of Path
        List of test structure files.
    """
    return sorted(
        [p for p in (Path(__file__).parent / "data" / "pdb").iterdir() if p.is_file()]
    )


def assemble_predictions(
    system_id: str,
) -> tuple[struc.AtomArray, struc.AtomArrayStack]:
    """
    Assemble test reference and model system structures for a given system ID.

    Parameters
    ----------
    system_id : str
        The system ID.

    Returns
    -------
    reference : AtomArray
        Reference structure.
    models : AtomArrayStack
        Model structures.
    """
    system_dir = Path(__file__).parent / "data" / "predictions" / system_id
    pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
    reference = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    pdbx_file = pdbx.CIFFile.read(system_dir / "models.cif")
    models = pdbx.get_structure(pdbx_file, model=None, include_bonds=True)
    return reference, models


def get_reference_metric(system_id: str, metric_name) -> np.ndarray[np.floating]:
    """
    Parse the reference metric computed by the legacy evaluation pipeline for the given
    system from ``ref_metrics.csv``.

    Parameters
    ----------
    system_id : str
        The system ID to get the metric for.
    metric_name : str
        The column name of the metric to get.

    Returns
    -------
    metric : np.ndarray, shape=(n_models, n_ligands) or shape=(n_models,), dtype=float
        The metric for the system.
        The ``n_models`` dimension specifies the metric value for each predicted model.
        The ``n_ligands`` dimension separates the metric value for each small molecule
        ligand, if applicable.
    """
    metrics_table = pd.read_csv(Path(__file__).parent / "data" / "ref_metrics.csv")
    metrics_table = metrics_table.loc[metrics_table["system_id"] == system_id]
    model_index = metrics_table["model_num"].to_numpy().astype(int)
    if metrics_table["ligand"].isna().all():
        metric = np.full((np.max(model_index) + 1, 1), np.nan)
        for i, value in zip(model_index, metrics_table[metric_name]):
            metric[i] = value
    else:
        molecule_index = np.array(
            [int(label.split("_")[1]) for label in metrics_table["ligand"]]
        )
        metric = np.full((np.max(model_index) + 1, np.max(molecule_index) + 1), np.nan)
        for i, j, value in zip(model_index, molecule_index, metrics_table[metric_name]):
            metric[i, j] = value
    return metric
