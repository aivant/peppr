"""
Based on https://github.com/cctbx/cctbx_project, governed on the following license:

cctbx.license.txt.
"""

__all__ = [
    "RotamerScore",
    "RamaScore",
    "get_fraction_of_rotamer_outliers",
    "get_fraction_of_rama_outliers",
]

import functools
import math
import pickle
import tarfile
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any
import numpy as np
import requests
from biotite import structure as struc
from biotite.structure import check_res_id_continuity
from scipy.interpolate import RegularGridInterpolator

_ROTAMER_BASE_DIR = Path.home() / ".local/share/top8000_lib"
_ROTAMER_DIR = _ROTAMER_BASE_DIR / "reference_data-master" / "Top8000"
_ROTA_OUTLIER_THRESHOLD = 0.003
_ROTA_ALLOWED_THRESHOLD = 0.02
_RAMA_FAVORED_THRESHOLD = 0.02
_RAMA_ALLOWED_THRESHOLD = 0.001
_RAMA_GENERAL_ALLOWED_THRESHOLD = 0.0005
_RAMA_CISPRO_ALLOWED_THRESHOLD = 0.0020


class RamaResidueType(StrEnum):
    """
    Enum for residue types used in the Ramachandran library.
    This enum defines the types of residues that can be used with the Ramachandran phi/psi grid data.
    """

    GLY = "gly"
    CISPRO = "cispro"
    TRANSPRO = "transpro"
    PREPRO = "prepro"
    ILEVAL = "ileval"
    GENERAL = "general"


class RotamerGridResidueMap(StrEnum):
    """
    Enum for residue names used in the rotamer (chi angles) grid data.
    This enum maps residue names to their corresponding tags used in the rotamer library.
    """

    ALA = "NA"
    GLY = "NA"  # ALA and GLY have no chi angles
    ARG = "arg"
    ASN = "asn"
    ASP = "asp"
    CYS = "cys"
    GLN = "gln"
    GLU = "glu"
    HIS = "his"
    ILE = "ile"
    LEU = "leu"
    LYS = "lys"
    MET = "met"
    PHE = "phetyr"
    PRO = "pro"
    SER = "ser"
    THR = "thr"
    TRP = "trp"
    TYR = "phetyr"
    VAL = "val"


class ConformerClass(IntEnum):
    """
    Enum for conformer classification types.

    This enum defines the possible classifications for different conformations [1]_.
    It can be one of the following:
        - "FAVORED": The residue is in a favored rotamer conformation.
        - "ALLOWED": The residue is in an allowed rotamer conformation.
        - "OUTLIER": The residue is in an outlier rotamer conformation.
        - "UNKNOWN": The residue's rotamer score could not be determined.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.25039
    """

    FAVORED = 1
    ALLOWED = 2
    OUTLIER = 3
    UNKNOWN = 4


@dataclass
class DihedralScore:
    """
    A dataclass to hold the results of rotamer and rama score checks.

    Attributes
    ----------
    pct : float
        The percentile score of the rotamer or rama check.
    classification : ConformerClass
        The classification of the score (e.g., FAVORED, ALLOWED, OUTLIER).

    References
    ----------
    .. [1] https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/
    """

    pct: float
    classification: ConformerClass


def _get_mmtbx_neg180_to_180_value(angle_deg: float) -> float:
    """
    Convert phi and psi angles to the range [-180, 180).

    Parameters
    ----------
    angle_deg : float
        The angle in degrees.

    Returns
    -------
    float
        The converted angle in the range [-180, 180).

    Notes
    -----
    Taken from https://github.com/cctbx/cctbx_project/blob/ee756cb47e375e11b4c37b65c418747f42104446/mmtbx/geometry_restraints/ramachandran.h#L230C7-L244C1
    """
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg <= -180:
        angle_deg += 360
    assert -180 <= angle_deg <= 180, (
        f"Angle must be in the range [-180, 180), but we got {angle_deg}"
    )
    return angle_deg


def _wrap_phi_psi(phi: float, psi: float) -> tuple[float, float]:
    """
    Wrap phi and psi angles to the range [-180, 180).

    Parameters
    ----------
    phi : float
        The phi angle in degrees.
    psi : float
        The psi angle in degrees.

    Returns
    -------
    tuple
        A tuple containing the converted phi and psi angles.

    Notes
    -----
    Taken from https://github.com/cctbx/cctbx_project/blob/ee756cb47e375e11b4c37b65c418747f42104446/mmtbx/geometry_restraints/ramachandran.h#L230C7-L244C1
    """
    return _get_mmtbx_neg180_to_180_value(phi), _get_mmtbx_neg180_to_180_value(psi)


def _wrap_symmetrical(
    residue_tag: RotamerGridResidueMap, wrapped_chis: list[float]
) -> list[float]:
    """
    Apply symmetrical wrapping to chi angles for specific amino acids.

    Parameters
    ----------
    residue_tag : RotamerGridResidueMap
        The residue tag enum
    wrapped_chis : list[float]
        The list of wrapped chi angles.

    Returns
    -------
    list[float]
        The list of symmetrically wrapped chi angles.

    Notes
    -----
    Taken from Source: https://github.com/cctbx/cctbx_project/blob/ee756cb47e375e11b4c37b65c418747f42104446/mmtbx/rotamer/rotamer_eval.py#L368
    """
    residue_tag = residue_tag.lower()
    if residue_tag == "asp" or residue_tag == "glu" or residue_tag == "phetyr":
        i = len(wrapped_chis) - 1
        if wrapped_chis[i] is not None:
            wrapped_chis[i] = wrapped_chis[i] % 180
            if wrapped_chis[i] < 0:
                wrapped_chis[i] += 180
    return wrapped_chis


def _wrap_chis(
    residue_tag: RotamerGridResidueMap, chis: list[float], symmetry: bool = True
) -> list[float]:
    """
    Wrap chi angles to the range [0, 360) or [-180, 180) depending on the residue type.

    Parameters
    ----------
    residue_tag : RotamerGridResidueMap
        The residue tag enum
    chis : list[float]
        The list of chi angles.
    symmetry : bool
        Whether to apply symmetrical wrapping.

    Returns
    -------
    list[float]
        The list of wrapped chi angles.

    Notes
    -----
    Taken from Source: https://github.com/cctbx/cctbx_project/blob/ee756cb47e375e11b4c37b65c418747f42104446/mmtbx/rotamer/rotamer_eval.py#L368
    """
    residue_tag = residue_tag.lower()
    wrapped_chis = []
    for i in range(0, len(chis)):
        if chis[i] is not None:
            wrapped_chis.append(chis[i] % 360)
            if wrapped_chis[i] < 0:
                wrapped_chis[i] += 360
        else:
            wrapped_chis.append(None)
    if symmetry:
        wrapped_chis = _wrap_symmetrical(residue_tag, wrapped_chis)
    return wrapped_chis


def _download_and_extract(url: str, dest_path: Path | None = None) -> Path:
    """
    Download a file from the given URL and extract it if it's a zip or tar archive.
    If `dest_path` is provided, it will be used as the destination file path.
    If `dest_path` is None, the file will be saved in the current directory with the same name as the URL.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    dest_path : Path
        The destination path where the file should be saved. If None, uses the URL's name.

    Returns
    -------
    Path
        The path where the file was saved or extracted.
    """
    if dest_path is None:
        dest_path = Path(Path(url).name)
    else:
        dest_path = Path(dest_path)
    # Ensure parent directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Download the file
    if dest_path.exists():
        return dest_path.parent
    r = requests.get(url)
    r.raise_for_status()
    data = r.content
    if dest_path:
        with open(dest_path, "wb") as f:
            f.write(data)
        # Check if it's a zip or tar file
        if str(dest_path).endswith(".zip"):
            with zipfile.ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(dest_path.parent)
            return dest_path.parent
        elif str(dest_path).endswith(".tar.gz") or str(dest_path).endswith(".tgz"):
            with tarfile.open(dest_path, "r:gz") as tar_ref:
                tar_ref.extractall(dest_path.parent)
            return dest_path.parent
    return dest_path.parent


def _get_residue_chis(
    atom_array: struc.AtomArray, res_id_mask: np.ndarray
) -> list[float]:
    """
    Compute the chi angles for a given residue.

    Parameters
    ----------
    atom_array : AtomArray
        The structure
    res_id_mask : np.ndarray
        A boolean mask selecting atoms of the residue (or an index for residue).

    Returns
    -------
    tuple[float]
        A tuple of chi angles (chi1, chi2, chi3, chi4). Missing chis will be omitted.

    Notes
    -----
    The chi angles are computed based on the standard definitions for each amino acid.
    In case of missing atoms, the corresponding chi angle will be omitted.
    """
    # We'll compute common chis by looking up named atoms for each residue type.
    res_atoms = atom_array[res_id_mask]
    # Build a map name -> coordinates
    coords = {
        name.strip(): coord for name, coord in zip(res_atoms.atom_name, res_atoms.coord)
    }
    # mapping of chi definitions per residue: set of atom name tuples (A,B,C,D) for chi1..chi4
    CHI_DEFS = {
        # Example: for ARG: chi1: N-CA-CB-CG  (names: 'N','CA','CB','CG' )
        "ARG": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"),
            ("CG", "CD", "NE", "CZ"),
        ],
        "LEU": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1"),
        ],  # LEU uses chi2 to CD1 or CD2 depending; this is simplified
        "VAL": [("N", "CA", "CB", "CG1")],
        "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
        "MET": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE"),
        ],
        "LYS": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"),
            ("CG", "CD", "CE", "NZ"),
        ],
        "PHE": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1"),
        ],
        "TRP": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1"),
        ],
        "TYR": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1"),
        ],
        "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
        "GLN": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1"),
        ],
        "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
        "GLU": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1"),
        ],
        "CYS": [("N", "CA", "CB", "SG")],
        "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
        "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
        "SER": [("N", "CA", "CB", "OG")],
        "THR": [("N", "CA", "CB", "OG1")],
        "ALA": [],  # ALA has no chi angles
        "GLY": [],  # GLY has no chi angles
    }
    # detect residue type name
    resname = res_atoms.res_name[0].upper()
    chis: list[float] = []
    if resname not in CHI_DEFS:
        return chis

    for i, atom_tuple in enumerate(CHI_DEFS[resname], start=1):
        try:
            p = [coords[a] for a in atom_tuple]
        except KeyError as e:
            warnings.warn(
                f"Skipping chi angle {i} for residue {resname}: {e} due to missing atoms"
            )
            # Skip if any atom is missing
            continue
        else:
            ang = math.degrees(struc.dihedral(*p))
            chis.append(ang)

    return chis


@functools.cache
def _download_top1000_lib() -> Path:
    """
    Download the Top8000 rotamer library from the Richardson Lab GitHub.
    This is a convenience function to download the entire library.

    Returns
    -------
    Path
        The path to the downloaded Top8000 rotamer library directory.
        If the directory already exists, it returns the existing path.
    """
    URL = "https://github.com/rlabduke/reference_data/archive/refs/heads/master.zip"
    if _ROTAMER_DIR.exists():
        return _ROTAMER_DIR
    data_dir = _download_and_extract(URL, _ROTAMER_BASE_DIR / "top1000.zip")
    if data_dir is None:
        raise RuntimeError("Failed to download or extract Top8000 rotamer library")
    return _ROTAMER_DIR


@functools.cache
def _load_contour_grid_text(resname_grid_data: Path | Any) -> dict[str, Any]:
    """
    Parse a Top8000 pct_contour_grid text file into a numpy grid and axis coordinate arrays.

    Parameters
    ----------
    resname_grid_data : Path | str
        The path to the grid data file or the raw text data.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the grid data, axes, and wrap information.
        ```{
        'grid': np.ndarray,
        'axes': [axis1_centers, axis2_centers, ...],
        'wrap': [True/False per axis]
        }```.

    Notes
    -----
    The grid data file should have a header with the number of dimensions and axis information,
    followed by the grid values. The format is expected to be:
    # number of dimensions: 3
    # x1: 0.0  360.0  36 true
    # x2: -180.0  180.0  36 true
    # x3: 0.0  360.0  36 false
    0.0 0.0 0.0 0.01
    The first line contains the number of dimensions, and each subsequent line describes an axis
    with its low and high bounds, number of bins, and whether it wraps around (true/false).
    If the file is not found or cannot be parsed, it returns None.
    If the file is a string, it will be treated as raw text data.
    If the file is a Path, it will be read from disk.
    If the file is not found or cannot be parsed, it returns None.
    """
    # Download data
    if isinstance(resname_grid_data, Path):
        with open(resname_grid_data, "r") as f:
            text = f.read()
    else:
        text = resname_grid_data
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Parse header
    axes_meta = []
    data_start_idx = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("# number of dimensions"):
            continue
        elif ln.strip().startswith("#   x"):
            parts = ln.split(":")[1].split()
            low, high, nbins, wrap = (
                float(parts[0]),
                float(parts[1]),
                int(parts[2]),
                parts[3].lower() == "true",
            )
            axes_meta.append((low, high, nbins, wrap))
        elif not ln.startswith("#"):
            data_start_idx = i
            break
    # Build coordinate arrays (bin centers)
    axes_coords = []
    wraps = []
    steps = []
    for low, high, nbins, wrap in axes_meta:
        step = (high - low) / nbins
        centers = np.linspace(low + step / 2, high - step / 2, nbins)
        axes_coords.append(centers)
        wraps.append(wrap)
        steps.append(step)

    # Prepare empty grid
    shape = [meta[2] for meta in axes_meta]
    grid = np.zeros(shape, dtype=float)

    # Fill grid from data lines
    for ln in lines[data_start_idx:]:
        parts = ln.split()
        coords = list(map(float, parts[:-1]))
        val = float(parts[-1])
        # Map coords to bin indices
        idxs = []
        for dim, (low, high, nbins, wrap) in enumerate(axes_meta):
            step = (high - low) / nbins
            bin_idx = (
                int((coords[dim] - low) / step) % nbins
                if wrap
                else int((coords[dim] - low) / step)
            )
            idxs.append(bin_idx)
        grid[tuple(idxs)] = val

    return {
        "grid": grid,
        "axes": axes_coords,
        "wrap": wraps,
        "span": np.array([am[:2] for am in axes_meta]),
        "steps": steps,
    }


@functools.cache
def _load_contour_grid_for_residue(
    grid_dirname: str,
    resname_tag: str,
    grid_data_tag: str = "rota8000",
    files_to_skip: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load contour grid for a given residue name from the Top8000 rotamer library.

    Parameters
    ----------
    grid_dirname : str
        The directory name where the contour grids are stored.
    resname_tag : str
        The residue name tag to load the contour grid for (e.g. "arg", "leu").
    grid_data_tag : str
        The grid data tag to load (default is "rota8000").
    files_to_skip : tuple[str, ...] | None
        A tuple of file names to skip when loading the grid data.

    Returns
    -------
    dict
        A dictionary containing the grid data, axes, and wrap information.
    """
    all_dict_contour = _load_all_contour_grid_from_pickle(
        grid_dirname=grid_dirname,
        grid_data_tag=grid_data_tag,
        files_to_skip=files_to_skip,
    )

    return all_dict_contour.get(resname_tag, {})


@functools.cache
def _generate_contour_grids_data(
    grid_dirname: str, grid_data_tag: str, files_to_skip: tuple[str, ...] | None = None
) -> None:
    """
    Generate contour grids data for the Top8000 rotamer library and save it to a pickle file.

    Parameters
    ----------
    grid_dirname : str
        The directory name where the contour grids will be saved.
    grid_data_tag : str
        The grid data tag to use (default is "rota8000").
    files_to_skip : tuple[str, ...] | None
        A tuple of file names to skip when generating the contour grids. If None, no files are skipped.

    Returns
    -------
    None
        This function saves the contour grids data to a pickle file.
        If the file already exists, it skips the generation.
    """
    output_path = _ROTAMER_DIR / f"{grid_dirname}.pkl"
    if output_path.exists():
        warnings.warn(
            f"Contour grids already generated at {output_path}. Skipping generation."
        )
        return
    data_dict = {}
    data_dir = _download_top1000_lib()
    if data_dir is None or not data_dir.exists():
        raise RuntimeError(
            "Top8000 rotamer library not found; cannot generate contour grids."
        )
    all_grid_files = list(data_dir.glob(f"{grid_dirname}/{grid_data_tag}-*.data"))

    if files_to_skip is not None:
        all_grid_files = [f for f in all_grid_files if f.name not in files_to_skip]

    if not all_grid_files:
        raise RuntimeError(f"No contour grid files found in {data_dir / grid_dirname}")
    for grid_file in all_grid_files:
        resname_tag = grid_file.stem.split("-")[
            1
        ].lower()  # e.g. "rota8000-arg.data" -> "arg"

        tmp_contour = _load_contour_grid_text(grid_file)
        if tmp_contour is None:
            continue
        data_dict[resname_tag] = tmp_contour
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)


@functools.cache
def _load_all_contour_grid_from_pickle(
    grid_dirname: str, grid_data_tag: str, files_to_skip: tuple[str, ...] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load all contour grids from a pickle file for the given grid directory and data tag.
    If the pickle file does not exist, it generates the contour grids data and saves it to a pickle file.

    Parameters
    ----------
    grid_dirname : str
        The directory name where the contour grids are stored.
    grid_data_tag : str
        The grid data tag to load (default is "rota8000").
    files_to_skip : tuple[str, ...] | None
        A tuple of file names to skip when loading the grid data. If None, no files are skipped.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        A dictionary containing the loaded contour grids.
    """
    contour_path = _ROTAMER_DIR / f"{grid_dirname}.pkl"
    if not contour_path.exists():
        _generate_contour_grids_data(
            grid_dirname=grid_dirname,
            grid_data_tag=grid_data_tag,
            files_to_skip=files_to_skip,
        )
    with open(contour_path, "rb") as f:
        data = pickle.load(f)
    return data


def _truncate_angles_within_a_step_of_grid_span(
    coord: float, low: float, high: float, step: float
) -> float:
    """
    Truncate an angle to be within a step of the grid span.

    This function ensures that the coordinate is within the specified low and high bounds,
    and if it is outside, it will return the closest bound if it is within one step of it.

    Parameters
    ----------
    coord : float
        The coordinate to truncate.
    low : float
        The lower bound of the grid span.
    high : float
        The upper bound of the grid span.
    step : float
        The step size of the grid.

    Returns
    -------
    float
        The truncated coordinate, which is either within the bounds or the closest bound if within one step.
    """
    # Handle edge cases
    if coord < low:
        if abs(coord - low) < step:
            return low
    elif coord > high:
        if abs(coord - high) < step:
            return high
    return coord


def _identify_gaps_in_atom_array(
    atom_array: struc.AtomArray,
) -> np.ndarray:
    """
    Identify gaps in the atom array based on missing atoms or residues.

    Parameters
    ----------
    atom_array : AtomArray
        The atom array to check for gaps.

    Returns
    -------
    np.ndarray
        An array of residue IDs where gaps are identified.
    """
    gap_ending_atom_idx = check_res_id_continuity(atom_array)
    # Add the preceeding atoms
    gap_both_atom_idx = np.concatenate([gap_ending_atom_idx, gap_ending_atom_idx - 1])
    return atom_array.res_id[gap_both_atom_idx]


def _interp_wrapped(
    resname_tag: str,
    grid_obj: dict[str, np.ndarray],
    coords_deg: list[float],
    coords_type: str,
) -> tuple[float, list[float]]:
    """
    Interpolate probability at given χ coords (deg), handling wrapping axes.

    Parameters
    ----------
    grid_obj : dict[str, np.ndarray]
        The grid object containing 'grid', 'axes', and 'wrap' information.
    coords_deg : list[float]
        List of coordinates in degrees, which could be [phi, psi] / [χ1, χ2, ...].
    coords_type : str
        The type of coordinates being interpolated (e.g., "phi-psi" or "chi").

    Returns
    -------
    tuple[float, list[float]]
        A tuple containing:
        - The interpolated value at the given coordinates.
        - The wrapped coordinates used for interpolation.
    """
    coords = []
    if coords_type not in ["phi-psi", "chi"]:
        raise ValueError(
            f"Invalid coords_type: {coords_type}. Expected 'phi-psi' or 'chi'."
        )
    if np.isnan(coords_deg).any():
        return np.nan, coords_deg

    if (coords_type == "phi-psi") and all(grid_obj["wrap"]):
        # Wrap phi and psi angles to [-180, 180)
        coords = list(_wrap_phi_psi(coords_deg[0], coords_deg[1]))

    elif (coords_type == "chi") and all(grid_obj["wrap"]):
        # Wrap chi angles to [0, 360) or [0, 180) depending on residue type
        coords = list(_wrap_chis(resname_tag, coords_deg, symmetry=True))

    coords = [
        _truncate_angles_within_a_step_of_grid_span(
            coord, grid_obj["axes"][i][0], grid_obj["axes"][i][-1], grid_obj["steps"][i]
        )
        for i, coord in enumerate(coords)
    ]

    interp_func = RegularGridInterpolator(
        tuple(grid_obj["axes"]), grid_obj["grid"], bounds_error=False, fill_value=np.nan
    )
    if interp_func(coords)[0] is np.nan:
        warnings.warn(
            f"Interpolated value is NaN for residue {resname_tag} at coords {coords}. Check if coords are within grid span {grid_obj['span']}."
        )

    return float(interp_func(coords)[0]), coords


def _check_rotamer(
    atom_array: struc.AtomArray, res_id: int, chain_id: str
) -> tuple[DihedralScore, list[float]]:
    """
    Check the rotamer classification for a given residue in the atom array.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The structure
    res_id : int
        The residue ID to check.
    chain_id : str
        The chain ID to check.

    Returns
    -------
    tuple[DihedralScore, list[float]]
        A tuple containing:
        - DihedralScore: The rotamer score and classification.
        - dict[str, float]: The observed chi angles for the residue.
    """
    mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id)
    if not np.any(mask):
        raise ValueError("Residue not found")

    resname = atom_array.res_name[mask][0].upper()
    observed = _get_residue_chis(atom_array, mask)

    if not observed:
        return DihedralScore(
            pct=np.nan, classification=ConformerClass.UNKNOWN
        ), observed

    resname_tag = RotamerGridResidueMap.__dict__.get(resname, None)

    if resname_tag is None or resname_tag == "NA":
        warnings.warn(f"Rotamer classification not available for residue {resname}.")
        return DihedralScore(
            pct=np.nan, classification=ConformerClass.UNKNOWN
        ), observed

    grid_obj = _load_contour_grid_for_residue(
        grid_dirname="Top8000_rotamer_pct_contour_grids",
        resname_tag=resname_tag,
        grid_data_tag="rota8000",
        files_to_skip=("rota8000-leu-raw.data",),  # Use rota8000-leu.data instead
    )

    # Build coords in same order as grid dims
    coords_list = []
    for i in range(len(grid_obj["axes"])):
        coords_list.append(observed[i] if i < len(observed) else 0.0)

    pct, wrapped_angles = _interp_wrapped(resname_tag, grid_obj, coords_list, "chi")

    if np.isnan(pct):
        classification = (
            ConformerClass.FAVORED
        )  # In mmtbx.validation.rotalyze, they counted this as FAVORED

    elif pct >= _ROTA_ALLOWED_THRESHOLD:
        classification = ConformerClass.FAVORED
    elif pct >= _ROTA_OUTLIER_THRESHOLD:
        classification = ConformerClass.ALLOWED
    else:
        classification = ConformerClass.OUTLIER

    return DihedralScore(pct=pct, classification=classification), wrapped_angles


@dataclass
class ResidueRotamerScore:
    """
    Rotamer score for a residue in a protein structure.

    This class represents the rotamer score for a single residue, including its name,
    ID, observed chi angles, rotamer score percentage, and classification.

    Attributes
    ----------
    resname : str
        The name of the residue (e.g. "ARG", "LEU").
    resid : int
        The ID of the residue in the structure.
    chain_id : str
        The chain ID of the residue.
    angles : dict[str, float]
        A dictionary containing the observed chi angles for the residue, e.g. {'chi1': 60.0, 'chi2': -45.0}.
    pct : float
        The rotamer score percentage for the residue, e.g. 0.01 for 1% favored.
    classification : ConformerClass
        The classification of the residue based on its rotamer score percentage.

    References
    ----------
    .. [1] https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/
    """

    res_name: str
    res_id: int
    chain_id: str
    angles: dict[str, float]
    pct: float
    classification: ConformerClass

    @staticmethod
    def from_residue(
        atom_array: struc.AtomArray, res_id: int, res_name: str, chain_id: str
    ) -> "ResidueRotamerScore":
        """
        Create ResidueRotamerScore from a residue object.

        Parameters
        ----------
        atom_array : struc.AtomArray
            The structure.
        res_id : int
            The residue ID to check.
        res_name : str
            The residue name to check (e.g. "ARG", "LEU").
        chain_id : str
            The chain ID to check.

        Returns
        -------
        ResidueRotamerScore
            An instance of ResidueRotamerScore containing the rotamer score for the residue.
        """
        score, angles = _check_rotamer(atom_array, res_id, chain_id)
        chis = {f"chi{i + 1}": ang for i, ang in enumerate(angles)}

        return ResidueRotamerScore(
            res_name=res_name,
            res_id=res_id,
            chain_id=chain_id,
            angles=chis,
            **asdict(score),
        )


def _classify_rotamers(atom_array: struc.AtomArray) -> list["ResidueRotamerScore"]:
    """
    Create ResidueRotamerScore from a protein structure.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The structure.

    Returns
    -------
    list[ResidueRotamerScore]
        An instance of RotamerScore containing the rotamer scores for the structure.
    """
    if all(atom_array.hetero):
        warnings.warn(
            "RotamerScore: Cannot be computed; atom_array must contain some protein residues"
        )
        return []
    atom_array = atom_array[
        struc.filter_canonical_amino_acids(atom_array) & ~atom_array.hetero
    ]
    if atom_array.array_length == 0:
        warnings.warn(
            "RotamerScore: Cannot be computed; atom_array must contain at least one residue"
        )
        return []
    rotamer_scores: list[ResidueRotamerScore] = []
    for chain_arr in struc.chain_iter(atom_array):
        chain_arr = chain_arr[struc.filter_canonical_amino_acids(chain_arr)]
        res_ids, res_names = struc.get_residues(chain_arr)
        rotamer_scores = []
        for res_id, res_name in zip(res_ids, res_names):
            chain_id = chain_arr.chain_id[chain_arr.res_id == res_id][0]

            if res_name not in RotamerGridResidueMap.__members__:
                continue
            rotamer_scores.append(
                ResidueRotamerScore.from_residue(
                    atom_array=atom_array,
                    res_id=res_id,
                    res_name=res_name,
                    chain_id=chain_id,
                )
            )

    return rotamer_scores


@dataclass
class RotamerScore:
    """
    Rotamer score for a given protein structure.
    This class represents the rotamer scores for all residues in a protein structure,
    including their names, IDs, observed chi angles, rotamer score percentages, and
    classifications.

    It is used to assess the quality of the protein structure based on the rotamer
    conformations and to identify potential outliers in the rotamer angles.

    Attributes
    ----------
    rotamer_scores : list[ResidueRotamerScore]
        A list of ResidueRotamerScore objects, each representing the rotamer score for a
        residue in the structure.
        Each ResidueRotamerScore contains the residue name, residue ID, chain ID,
        observed chi angles, rotamer score percentage and classification.

    References
    ----------
    .. [1] http://molprobity.biochem.duke.edu/help/validation_options/summary_table_guide.html
    .. [2] https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/
    """

    rotamer_scores: list[ResidueRotamerScore] = field(
        default_factory=list,
        metadata={
            "description": "List of rotamer scores for each residue in the structure."
        },
    )

    @staticmethod
    def from_atoms(atom_array: struc.AtomArray) -> "RotamerScore":
        """
        Create RotamerScore from a protein structure or a stack of structures.

        Parameters
        ----------
        atom_array : struc.AtomArray | struc.AtomArrayStack
            The AtomArray or AtomArrayStack containing the structure(s).

        Returns
        -------
        RotamerScore
            An instance of RotamerScore containing the rotamer scores for the structure(s).
        """
        return RotamerScore(rotamer_scores=_classify_rotamers(atom_array))


def _get_residue_phi_psi_omega(
    atom_array: struc.AtomArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the phi, psi, and omega dihedral angles for a given residue.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The structure.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the phi, psi, and omega angles in degrees.
    """
    if atom_array.shape[0] < 4:
        raise ValueError(
            "atom_array must contain at least 4 atoms for dihedral angle calculation"
        )
    phi, psi, omega = struc.dihedral_backbone(atom_array)
    # Remove invalid values (NaN) at first and last position
    phi = phi[1:-1]
    psi = psi[1:-1]
    omega = omega[1:-1]

    # Convert to degrees, keeping NaNs as NaNs
    phi = np.where(np.isnan(phi), np.nan, np.rad2deg(phi))
    psi = np.where(np.isnan(psi), np.nan, np.rad2deg(psi))
    omega = np.where(np.isnan(omega), np.nan, np.rad2deg(omega))
    return phi, psi, omega


def _is_cislike_peptide(omega: float) -> bool:
    """
    Check if a peptide bond is cis-like based on the omega dihedral angle.
    A peptide bond is considered cis-like if the omega angle is between -30 and 30 degrees.
    This is based on the definition used in the mmtbx.validation.omegalyze module [1]_.

    Parameters
    ----------
    omega : float
        The omega dihedral angle in degrees.

    Returns
    -------
    bool
        True if the peptide bond is cis-like (omega between -90 and 90 degrees), False otherwise.

    References
    ----------
    .. [1] hhttps://github.com/cctbx/cctbx_project/blob/3de736be359daeba2a2f3d7da6a4a94faf4ec4d5/mmtbx/validation/omegalyze.py#L560C1-L564C20
    """
    if np.isnan(omega):
        return False
    if (omega > -30) and (omega < 30):
        return True
    return False


def _assign_rama_types(
    atom_array: struc.AtomArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Assign Ramachandran types to residues based on their phi, psi, and omega angles.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The structure.

    Returns
    -------
    dict
        A dictionary containing the phi, psi, omega angles and their corresponding
        Ramachandran types.

    Notes
    -----
    This function assumes `atom_array` is already cleaned and contains only relevant
    residues.
    It should not contain water or other non-protein residues.
    """
    # TODO: Make this function more performant by avoiding multiple calls to struc.dihedral_backbone
    phis, psis, omegas = _get_residue_phi_psi_omega(atom_array)

    _, res_names = struc.get_residues(atom_array)
    next_res_names = np.roll(res_names, -1)[
        1:-1
    ]  # Shifted by -1, remove first and last

    def rama_type(res_name: str, next_res_name: str, omega_val: float) -> str:
        if res_name == "GLY":
            return RamaResidueType.GLY
        elif res_name == "PRO":
            return (
                RamaResidueType.CISPRO
                if _is_cislike_peptide(omega_val)
                else RamaResidueType.TRANSPRO
            )
        elif next_res_name == "PRO":
            return RamaResidueType.PREPRO
        elif res_name in ["ILE", "VAL"]:
            return RamaResidueType.ILEVAL
        else:
            return RamaResidueType.GENERAL

    rama_types = [
        rama_type(res_name, next_res_name, omega_val)
        for res_name, next_res_name, omega_val in zip(
            res_names[1:-1], next_res_names, omegas
        )
    ]

    return phis, psis, omegas, rama_types


def _check_rama(
    phi: float,
    psi: float,
    resname_tag: RamaResidueType,
) -> "DihedralScore":
    """
    Check the Ramachandran classification for a given residue based on its phi, psi, and omega angles.

    Parameters
    ----------
    phi : float
        The phi angle in degrees.
    psi : float
        The psi angle in degrees.
    resname_tag : RamaResidueType
        The residue name tag enum

    Returns
    -------
    DihedralScore
        An instance of DihedralScore containing the Ramachandran score and classification.

    Notes
    -----
    This function uses the Top8000 Ramachandran contour grids to classify the residue.
    It checks the phi, psi, and omega angles against the grids and assigns a
    classification of "FAVORED", "ALLOWED", or "OUTLIER" based on the percentage of
    favored conformations.
    If the residue is CIS or PRO, it uses special handling for those residues.
    The thresholds for classification are defined as follows:
    - FAVORED: pct >= `_RAMA_FAVORED_THRESHOLD`
    - ALLOWED: pct >= `_RAMA_ALLOWED_THRESHOLD` (or `_RAMA_GENERAL_ALLOWED_THRESHOLD`
      for "general" residues, or `_RAMA_CISPRO_ALLOWED_THRESHOLD` for "cispro" residues)
    - OUTLIER: pct < `_RAMA_ALLOWED_THRESHOLD` (or `_RAMA_GENERAL_ALLOWED_THRESHOLD` for
      "general" residues, or `_RAMA_CISPRO_ALLOWED_THRESHOLD` for "cispro" residues)

    References
    ----------
    .. [1] https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/
    """

    grid_obj = _load_contour_grid_for_residue(
        grid_dirname="Top8000_ramachandran_pct_contour_grids",
        resname_tag=resname_tag,
        grid_data_tag="rama8000",
        files_to_skip=None,
    )
    # Build coords in same order as grid dims
    coords_list = [phi, psi]
    pct, _ = _interp_wrapped(resname_tag, grid_obj, coords_list, "phi-psi")
    if resname_tag == "general":
        if pct >= _RAMA_GENERAL_ALLOWED_THRESHOLD:
            classification = ConformerClass.ALLOWED
        else:
            classification = ConformerClass.OUTLIER
    elif resname_tag == "cispro":
        if pct >= _RAMA_CISPRO_ALLOWED_THRESHOLD:
            classification = ConformerClass.ALLOWED
        else:
            classification = ConformerClass.OUTLIER
    else:
        if pct >= _RAMA_ALLOWED_THRESHOLD:
            classification = ConformerClass.ALLOWED
        else:
            classification = ConformerClass.OUTLIER
    if pct >= _RAMA_FAVORED_THRESHOLD:
        classification = ConformerClass.FAVORED

    return DihedralScore(pct=pct, classification=classification)


@dataclass
class ResidueRamaScore:
    """
    Ramachandran score for a residue in a protein structure.
    This class represents the Ramachandran score for a single residue, including its
    name, ID, observed phi, psi, omega angles, Ramachandran score percentage and
    classification.
    """

    res_name: str
    res_id: int
    chain_id: str
    resname_tag: str
    angles: dict[str, float]
    pct: float
    classification: ConformerClass

    @staticmethod
    def from_phi_psi_omega(
        phi: float,
        psi: float,
        omega: float,
        res_id: int,
        res_name: str,
        chain_id: str,
        resname_tag: str,
    ) -> "ResidueRamaScore":
        result = _check_rama(phi, psi, resname_tag)
        return ResidueRamaScore(
            res_id=res_id,
            res_name=res_name,
            resname_tag=resname_tag,
            chain_id=chain_id,
            **asdict(result),
            angles={"phi": phi, "psi": psi, "omega": omega},
        )


def _classify_phi_psi(
    atom_array: struc.AtomArray,
) -> list["ResidueRamaScore"]:
    """
    Create RamaScore from a protein structure.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The structure.

    Returns
    -------
    list[RamaScore]
        An instance of RamaScore containing the Ramachandran scores for the structure.
    """
    if all(atom_array.hetero):
        warnings.warn(
            "RotamerScore: Cannot be computed; atom_array must contain some protein residues"
        )
        return []
    atom_array = atom_array[
        struc.filter_canonical_amino_acids(atom_array) & ~atom_array.hetero
    ]
    if atom_array.array_length == 0:
        warnings.warn(
            "RamaScore: Cannot be computed; atom_array must contain at least one residue"
        )
        return []

    rama_scores = []
    for chain_id in struc.get_chains(atom_array):
        chain_arr = atom_array[atom_array.chain_id == chain_id]
        phis, psis, omegas, rama_types = _assign_rama_types(chain_arr)
        res_ids, res_names = struc.get_residues(chain_arr)
        # Identify breakpoints in backbone chains; this is to be skipped as
        # their scores are not reliable
        gaps = _identify_gaps_in_atom_array(chain_arr)

        for phi, psi, omega, res_id, res_name, rama_type in zip(
            phis,
            psis,
            omegas,
            res_ids[1:-1],
            res_names[1:-1],
            rama_types,
        ):
            if res_id in gaps:
                # Skip residues with gaps in the backbone chain
                continue
            if res_name not in RotamerGridResidueMap.__members__:
                # Skip residues that are not in the grid map
                continue

            rama_scores.append(
                ResidueRamaScore.from_phi_psi_omega(
                    phi=phi,
                    psi=psi,
                    omega=omega,
                    res_id=res_id,
                    res_name=res_name,
                    chain_id=atom_array.chain_id[atom_array.res_id == res_id][0],
                    resname_tag=rama_type,
                )
            )

    return rama_scores


@dataclass
class RamaScore:
    """
    Ramachandran score for a given protein structure.
    This class represents calculates the likelihood of each residue's phi/psi angles to occur in crystallized structures
    based on the Top8000 Ramachandran contour grids. It includes their names, IDs, observed phi, psi, omega angles,
    Ramachandran score percentages, and classifications. The thresholsds for classification are defined as follows:
    - FAVORED: pct >= `_RAMA_FAVORED_THRESHOLD`
    - ALLOWED: pct >= `_RAMA_ALLOWED_THRESHOLD` (or `_RAMA_GENERAL_ALLOWED_THRESHOLD` for "general" residues,
    or `_RAMA_CISPRO_ALLOWED_THRESHOLD` for "cispro" residues)
    - OUTLIER: pct < `_RAMA_ALLOWED_THRESHOLD` (or `_RAMA_GENERAL_ALLOWED_THRESHOLD` for "general" residues,
    or `_RAMA_CISPRO_ALLOWED_THRESHOLD` for "cispro" residues)

    It is used to assess the quality of the protein structure based on the Ramachandran angles
    and to identify potential outliers in the phi/psi angles..

    Attributes
    ----------
    rama_scores : list[ResidueRamaScore]
        A list of ResidueRamaScore objects, each representing the Ramachandran score for a residue in the structure.

    References
    ----------
    .. [1] https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/
    """

    rama_scores: list[ResidueRamaScore] = field(
        default_factory=list,
        metadata={
            "description": "List of Ramachandran scores for each residue in the structure."
        },
    )

    @staticmethod
    def from_atoms(atom_array: struc.AtomArray) -> "RamaScore":
        """
        Create RamaScore from a protein structure.

        Parameters
        ----------
        atom_array : struc.AtomArray
            The structure.

        Returns
        -------
        RamaScore
            An instance of RamaScore containing the Ramachandran scores for the
            structure.
        """
        return RamaScore(rama_scores=_classify_phi_psi(atom_array))


def get_fraction_of_rotamer_outliers(atom_array: struc.AtomArray) -> float:
    """
    Compute the fraction of rotamer outliers for given structure.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The structure.

    Returns
    -------
    float
        The fraction of rotamer outliers in the structure, calculated as the number of outlier
        rotamers divided by the total number of rotamers.
    """
    outlier_rotamers = 0
    total_rotamers = 0
    result = RotamerScore.from_atoms(atom_array)
    for rotamer_score in result.rotamer_scores:
        if rotamer_score.classification == ConformerClass.OUTLIER:
            outlier_rotamers += 1
        total_rotamers += 1
    return outlier_rotamers / total_rotamers if total_rotamers > 0 else 0.0


def get_fraction_of_rama_outliers(
    atom_array: struc.AtomArray,
) -> float:
    """
    Compute the fraction of Ramachandran outliers for given structure.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The AtomArray or AtomArrayStack containing the structure(s).

    Returns
    -------
    float
        The fraction of Ramachandran outliers in the structure, calculated as the number of outlier
        Ramachandran angles divided by the total number of Ramachandran angles.
    """
    outlier_rama = 0
    total_rama = 0
    result = RamaScore.from_atoms(atom_array)

    for rama_score in result.rama_scores:
        if rama_score.classification == ConformerClass.OUTLIER:
            outlier_rama += 1
        total_rama += 1
    return outlier_rama / total_rama if total_rama > 0 else 0.0
