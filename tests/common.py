from pathlib import Path
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd


def list_test_predictions() -> list[str]:
    """
    List all system IDs for test systems with predictions.

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
    Assemble test reference and pose system structures for a given system ID.

    Parameters
    ----------
    system_id : str
        The system ID.

    Returns
    -------
    reference : AtomArray
        Reference structure.
    poses : AtomArrayStack
        Pose structures.
    """
    system_dir = Path(__file__).parent / "data" / "predictions" / system_id
    pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
    reference = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    poses = []
    for pose_path in sorted(system_dir.glob("poses/*.cif")):
        pdbx_file = pdbx.CIFFile.read(pose_path)
        poses.append(pdbx.get_structure(pdbx_file, model=1, include_bonds=True))
    return reference, struc.stack(poses)


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
    metric : np.ndarray, shape=(n_poses, n_ligands) or shape=(n_poses,), dtype=float
        The metric for the system.
        The ``n_poses`` dimension specifies the metric value for each predicted pose.
        The ``n_ligands`` dimension separates the metric value for each small molecule
        ligand, if applicable.
    """
    metrics_table = pd.read_csv(Path(__file__).parent / "data" / "ref_metrics.csv")
    metrics_table = metrics_table.loc[metrics_table["system_id"] == system_id]
    pose_index = metrics_table["pose_num"].to_numpy().astype(int)
    molecule_index = metrics_table["ligand_num"] - 1
    metric = np.full((np.max(pose_index) + 1, np.max(molecule_index) + 1), np.nan)
    for i, j, value in zip(pose_index, molecule_index, metrics_table[metric_name]):
        metric[i, j] = value
    return metric
