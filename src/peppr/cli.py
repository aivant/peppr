import glob
import json
import pickle
import sys
from io import FileIO
from pathlib import Path
import biotite.structure as struc
import biotite.structure.io.mol as mol
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import click
import pandas as pd
from peppr.evaluator import Evaluator
from peppr.metric import *
from peppr.selector import *
from peppr.version import __version__

_METRICS = {
    "monomer-rmsd": MonomerRMSD(2.0),
    "monomer-tm-score": MonomerTMScore(),
    "monomer-lddt": MonomerLDDTScore(),
    "ligand-lddt": IntraLigandLDDTScore(),
    "lddt-pli": LDDTPLIScore(),
    "lddt-ppi": LDDTPPIScore(),
    "global-lddt": GlobalLDDTScore(),
    "dockq": DockQScore(),
    "dockq-ppi": DockQScore(include_pli=False),
    "lrmsd": LigandRMSD(),
    "irmsd": InterfaceRMSD(),
    "fnat": ContactFraction(),
    "pocket-lrmsd": PocketAlignedLigandRMSD(),
    "bisy-rmsd": BiSyRMSD(2.0),
    "bond-length-violations": BondLengthViolations(),
    "clash-count": ClashCount(),
}


@click.group(
    help="It's a package for evaluation of predicted poses, right?",
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.version_option(__version__)
def cli() -> None:
    pass


@cli.command()
@click.argument("EVALUATOR", type=click.File("wb", lazy=True))
@click.argument("METRIC", type=click.Choice(_METRICS.keys()), nargs=-1, required=True)
def create(evaluator: click.File, metric: tuple[str, ...]) -> None:
    """
    Initialize a new peppr evaluation.

    The peppr.pkl file tracking the future evaluation will be written to the given
    EVALUATOR file.
    The metrics that should be computed are given by the METRIC arguments.
    """
    metrics = [_METRICS[m] for m in metric]
    ev = Evaluator(metrics, tolerate_exceptions=True)
    _evaluator_to_file(evaluator, ev)


@cli.command()
@click.option(
    "--id",
    "-i",
    type=str,
    help="The system ID, by default taken from the reference file name",
)
@click.argument("EVALUATOR", type=click.File("rb+", lazy=True))
@click.argument(
    "REFERENCE", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "POSE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
    required=True,
)
def evaluate(
    evaluator: click.File, reference: Path, pose: tuple[Path, ...], id: str | None
) -> None:
    """
    Evaluate a single system.

    Run the metrics defined by the EVALUATOR on the given system, defined by the
    REFERENCE path and POSE paths, and store the results in the EVALUATOR file.
    """
    ev = _evaluator_from_file(evaluator)
    ref = _load_system(reference)
    poses = [_load_system(path) for path in pose]
    ev.feed(id or reference.stem, ref, poses)
    _evaluator_to_file(evaluator, ev)


@cli.command()
@click.argument("EVALUATOR", type=click.File("rb+", lazy=True))
@click.argument("REFERENCE", type=str)
@click.argument("POSE", type=str)
def evaluate_batch(evaluator: click.File, reference: str, pose: str) -> None:
    """
    Evaluate multiple systems.

    Run the metrics defined by the EVALUATOR on the given systems, defined by the
    REFERENCE and POSE glob patterns, and store the results in the
    EVALUATOR file.
    If multiple poses should be evaluated for a system, POSE must be a pattern
    that matches directories of files instead of a single file.
    Note that REFERENCE and POSE are assigned to each other in
    lexicographical order.
    """
    ev = _evaluator_from_file(evaluator)
    reference_paths = sorted(
        [Path(path) for path in glob.glob(reference, recursive=True)]
    )
    pose_paths = sorted([Path(path) for path in glob.glob(pose, recursive=True)])
    if len(reference_paths) != len(pose_paths):
        raise click.UsageError(
            f"Number of reference files ({len(reference_paths)}) "
            f"does not match the number of pose files ({len(pose_paths)})"
        )

    system_ids = _find_unique_part(reference_paths)
    # Potentially remove the file suffix from the system ID
    for i, system_id in enumerate(system_ids):
        splitted = system_id.split(".")
        if len(splitted) > 1 and splitted[-1] in ["cif", "bcif", "pdb", "mol", "sdf"]:
            system_ids[i] = splitted[0]

    for system_id, ref_path, pose_path in zip(system_ids, reference_paths, pose_paths):
        if ref_path.is_dir():
            raise click.UsageError(
                "REFERENCE glob pattern must point to files, but found a directory"
            )
        reference = _load_system(ref_path)
        if pose_path.is_dir():
            poses = [
                _load_system(path)
                for path in sorted(pose_path.iterdir())
                if path.is_file()
            ]
        else:
            poses = _load_system(pose_path)
        ev.feed(system_id, reference, poses)
    _evaluator_to_file(evaluator, ev)


@cli.command()
@click.argument("EVALUATOR", type=click.File("rb", lazy=True))
@click.argument("TABLE", type=click.Path(exists=False, path_type=Path))
@click.argument("SELECTOR", type=str, nargs=-1)
def tabulate(evaluator: click.File, table: Path, selector: tuple[str, ...]) -> None:
    """
    Tabulate metric results for each system.

    Read the EVALUATOR file and write a table of the metrics for each system to the
    given TABLE CSV file.
    For systems with multiple poses, the metric results are selected by the given
    SELECTOR (may be multiple).
    Supported SELECTOR values are: 'mean', 'median', 'oracle', 'top<n>'.
    """
    ev = _evaluator_from_file(evaluator)
    df = pd.DataFrame(
        ev.tabulate_metrics(selectors=[_create_selector(sel) for sel in selector])
    )
    df.to_csv(table, index_label="System ID")


@cli.command()
@click.argument("EVALUATOR", type=click.File("rb", lazy=True))
@click.argument("SUMMARY", type=click.File("w", lazy=True))
@click.argument("SELECTOR", type=str, nargs=-1)
def summarize(
    evaluator: click.File, summary: click.File, selector: tuple[str, ...]
) -> None:
    """
    Aggregate metrics over all systems.

    Read the EVALUATOR file and write a summary of the metrics aggregated over all
    systems to the given SUMMARY .json file.
    For systems with multiple poses, the metric results are selected by the given
    SELECTOR (may be multiple).
    Supported SELECTOR values are: 'mean', 'median', 'oracle', 'top<n>'.
    """
    ev = _evaluator_from_file(evaluator)
    data = ev.summarize_metrics(selectors=[_create_selector(sel) for sel in selector])
    json.dump(data, summary, indent=2)


@cli.command()
@click.argument("METRIC", type=click.Choice(_METRICS.keys()))
@click.argument(
    "REFERENCE", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("POSE", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def run(metric: str, reference: Path, pose: Path) -> None:
    """
    Compute a single metric for the given system.

    The given METRIC is run on the given REFERENCE and POSE and the result is
    written to STDOUT.
    """
    metric = _METRICS[metric]
    reference = _load_system(reference)
    pose = _load_system(pose)
    result = metric.evaluate(reference, pose)
    print(f"{result:.3f}", file=sys.stdout)


def _evaluator_from_file(file: FileIO) -> Evaluator:
    """
    Load a :class:`Evaluator` from the pickle representation in the given file.

    Parameters
    ----------
    file : file-like
        The file to read the pickled evaluator from.

    Returns
    -------
    Evaluator
        The evaluator.
    """
    return pickle.load(file)


def _evaluator_to_file(file: FileIO, evaluator: Evaluator) -> None:
    """
    Pickle the given :class:`Evaluator` and write it to the given file.

    Parameters
    ----------
    file : file-like
        The file to write the pickled evaluator to.
    evaluator : Evaluator
        The evaluator to pickle.
    """
    file.seek(0)
    pickle.dump(evaluator, file)


def _create_selector(selector_string: str) -> Selector:
    """
    Create a :class:`Selector` object from a string representation.

    Parameters
    ----------
    selector_string : {'mean', 'median', 'oracle', 'top<n>' and 'random<n>'}
        The string representation of the selector.

    Returns
    -------
    Selector
        The selector.
    """
    if selector_string == "mean":
        return MeanSelector()
    elif selector_string == "median":
        return MedianSelector()
    elif selector_string == "oracle":
        return OracleSelector()
    elif selector_string.startswith("top"):
        return TopSelector(int(selector_string[3:]))
    elif selector_string.startswith("random"):
        return RandomSelector(int(selector_string[6:]))
    else:
        raise click.BadParameter(f"Selector '{selector_string}' is not supported")


def _load_system(path: Path) -> struc.AtomArray:
    """
    Load a structure from a a variety of file formats.

    Parameters
    ----------
    path : Path
        The path to the structure file.
        The format is determined by the file extension.

    Returns
    -------
    AtomArray
        The system.
    """
    try:
        match path.suffix:
            case ".cif" | ".mmcif" | ".pdbx":
                cif_file = pdbx.CIFFile.read(path)
                return pdbx.get_structure(cif_file, model=1, include_bonds=True)
            case ".bcif":
                bcif_file = pdbx.BinaryCIFFile.read(path)
                return pdbx.get_structure(bcif_file, model=1, include_bonds=True)
            case ".pdb":
                pdb_file = pdb.PDBFile.read(path)
                return pdb.get_structure(pdb_file, model=1, include_bonds=True)
            case ".mol" | ".sdf":
                if path.suffix == ".sdf":
                    ctab_file = mol.SDFile.read(path)
                else:
                    ctab_file = mol.MOLFile.read(path)
                system = mol.get_structure(ctab_file)
                system.hetero[:] = True
                system.res_name[:] = "LIG"
                system.atom_name = struc.create_atom_names(system)
                return system
            case _:
                raise click.BadParameter(f"Unsupported file format '{path.suffix}'")
    except Exception as e:
        raise click.FileError(path.as_posix(), hint=str(e))


def _find_unique_part(paths: list[Path]) -> list[str]:
    """
    Find the last component of the given paths that is unique across all paths.

    Parameters
    ----------
    paths : list of Path
        The path to get the unique component from.

    Returns
    -------
    list of str
        for each path in `paths` the unique component.
    """
    # Iterate from the end to the beginning
    components = [path.parts[::-1] for path in paths]
    for i in range(max([len(c) for c in components])):
        component_of_each_path = [path_comp[i] for path_comp in components]
        if len(set(component_of_each_path)) == len(component_of_each_path):
            return component_of_each_path
    raise click.UsageError(
        "No unique system ID could be parsed from the given glob pattern"
    )
