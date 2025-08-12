"""
Backbone-dependent and backbone-independent rotamer checking using
Richardson Lab Top8000 rotamer library.
https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/

This is the library useful programs like MolProbity's "rotamer" validation, which uses
mmtbx.rotamer.rotamer_eval and mmtbx.validation.rotalyze and  mmtbx.validation.ramalyze under the hood.
"""

__all__ = [
    "RotamerScore",
    "ResidueRotamerScore",
    "ResidueRamaScore",
    "RamaScore",
    "get_fraction_of_rotamer_outliers",
    "get_fraction_of_rama_outliers",
]

import functools
import logging
import math
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import requests
from biotite import structure as struc
from pydantic import BaseModel, ConfigDict, Field
from scipy.interpolate import RegularGridInterpolator

# Configure basic logging to the console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOG = logging.getLogger(__name__)

ROTA_OUTLIER_THRESHOLD = 0.003
ROTA_ALLOWED_THRESHOLD = 0.02
RAMA_FAVORED_THRESHOLD = 0.02
RAMA_ALLOWED_THRESHOLD = 0.001
RAMA_GENERAL_ALLOWED_THRESHOLD = 0.0005
RAMA_CISPRO_ALLOWED_THRESHOLD = 0.0020


grid_residue_map = {
    "ALA": None,
    "GLY": None,  # ALA and GLY have no chi angles
    "ARG": "arg",
    "ASN": "asn",
    "ASP": "asp",
    "CYS": "cys",
    "GLN": "gln",
    "GLU": "glu",
    "HIS": "his",
    "ILE": "ile",
    "LEU": "leu",
    "LYS": "lys",
    "MET": "met",
    "PHE": "phetyr",
    "PRO": "pro3d",
    "SER": "ser",
    "THR": "thr",
    "TRP": "trp",
    "TYR": "phetyr",
    "VAL": "val",
}

grid_axex_span = {
    "ARG": "arg",
    "ASN": "asn",
    "ASP": "asp",
    "CYS": "cys",
    "GLN": "gln",
    "GLU": "glu",
    "HIS": "his",
    "ILE": "ile",
    "LEU": "leu",
    "LYS": "lys",
    "MET": "met",
    "PHE": "phetyr",
    "PRO": "pro3d",
    "SER": "ser",
    "THR": "thr",
    "TRP": "trp",
    "TYR": "phetyr",
    "VAL": "val",
}
_ROTAMER_BASE_DIR = Path.home() / ".local/share/top8000_lib"
_ROTAMER_DIR = _ROTAMER_BASE_DIR / "reference_data-master" / "Top8000"


def wrap_angle(angle_deg: float, mode: str = "±180") -> float:
    """
    Wrap an angle (in degrees) to a specified range.

    Parameters
    ----------
    angle_deg : float or array-like
        Input angle(s) in degrees.
    mode : str
        Wrapping mode:
        - "±180"   : range [-180, 180)
        - "0-360"  : range [0, 360)
        - "0-180"  : range [0, 180] (folded)

    Returns
    -------
    float
        Wrapped angle(s) in the chosen range.
    """

    if mode == "±180":
        return ((angle_deg + 180.0) % 360.0) - 180.0

    elif mode == "0-360":
        return angle_deg % 360.0

    elif mode == "0-180":
        a = angle_deg % 360.0
        if a > 180.0:
            return a - 180.0
        else:
            return abs(a)

    else:
        raise ValueError(
            f"Unknown mode '{mode}', choose from '±180', '0-360', '0-180'."
        )


def download_and_extract(url: str, dest_path: Path | None = None) -> Path | None:
    """
    Download a file from the given URL and extract it if it's a zip or tar archive.
    If `dest_path` is provided, it will be used as the destination file path.
    If `dest_path` is None, the file will be saved in the current directory with the same name as the URL.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    dest_path : Path | None
        The destination path where the file should be saved. If None, uses the URL's name.

    Returns
    -------
    Path | None
        The path where the file was saved or extracted.
    """
    if dest_path is None:
        dest_path = Path(Path(url).name)
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
            import zipfile

            with zipfile.ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(dest_path.parent)
            return dest_path.parent
        elif str(dest_path).endswith(".tar.gz") or str(dest_path).endswith(".tgz"):
            import tarfile

            with tarfile.open(dest_path, "r:gz") as tar_ref:
                tar_ref.extractall(dest_path.parent)
            return dest_path.parent
    return None


def get_residue_chis(
    atom_array: struc.AtomArray, res_id_mask: np.ndarray
) -> dict[str, float]:
    """
    Compute the chi angles for a given residue.

    Parameters
    ----------
    atom_array : AtomArray
        The AtomArray containing the structure.
    res_id_mask : np.ndarray
        A boolean mask selecting atoms of the residue (or an index for residue).

    Returns
    -------
    dict
        A dictionary with chi angles, e.g. {'chi1': angle_deg, 'chi2': angle_deg, ...}.
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
            ("CB", "CG", "CD1", "CE1"),
        ],
        "TRP": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1"),
            ("CB", "CG", "CD1", "CE2"),
        ],
        "TYR": [
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1"),
            ("CB", "CG", "CD1", "CE1"),
        ],
        "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
        "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OE1")],
        "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
        "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OE1")],
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
    chis: dict[str, float] = {}
    if resname not in CHI_DEFS:
        return chis

    for i, atom_tuple in enumerate(CHI_DEFS[resname], start=1):
        try:
            p = [coords[a] for a in atom_tuple]
            ang = math.degrees(struc.dihedral(*p))
            chis[f"chi{i}"] = ang
        except Exception as e:
            LOG.warning(f"Failed to compute chi angle {i} for residue {resname}: {e}")
            # Skip if any atom is missing
            pass
    return chis


@functools.lru_cache
def download_top1000_lib() -> Path:
    """
    Download the Top8000 rotamer library from the Richardson Lab GitHub.
    This is a convenience function to download the entire library.

    Returns
    -------
    Path
        The path to the downloaded Top8000 rotamer library directory.
        If the directory already exists, it returns the existing path.
    """
    if _ROTAMER_DIR.exists():
        return _ROTAMER_DIR
    url = "https://github.com/rlabduke/reference_data/archive/refs/heads/master.zip"
    data_dir = download_and_extract(url, _ROTAMER_BASE_DIR / "top1000.zip")
    if data_dir is None:
        raise RuntimeError("Failed to download or extract Top8000 rotamer library")
    LOG.info(f"Top8000 rotamer library downloaded and extracted to {data_dir}")
    return _ROTAMER_DIR


@functools.lru_cache
def load_contour_grid_text(resname_grid_data: Path | Any) -> dict[str, Any]:
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
        {
        'grid': np.ndarray,
        'axes': [axis1_centers, axis2_centers, ...],
        'wrap': [True/False per axis]
        }.

    Notes
    -----
    The grid data file should have a header with the number of dimensions and axis information,
    followed by the grid values. The format is expected to be:
    # number of dimensions: 3
    #   x1: 0.0  360.0  36 true
    #   x2: -180.0  180.0  36 true
    #   x3: 0.0  360.0  36 false
    #   0.0 0.0 0.0 0.01
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
            # Format: "#   x1: 0.0  360.0  36 true"
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
    for low, high, nbins, wrap in axes_meta:
        step = (high - low) / nbins
        centers = np.linspace(low + step / 2, high - step / 2, nbins)
        axes_coords.append(centers)
        wraps.append(wrap)

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
    }


@functools.lru_cache
def load_contour_grid_for_residue(
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
    all_dict_contour = load_all_contour_grid_from_pickle(
        grid_dirname=grid_dirname,
        grid_data_tag=grid_data_tag,
        files_to_skip=files_to_skip,
    )

    return all_dict_contour.get(resname_tag, {})


@functools.lru_cache
def generate_contour_grids_data(
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
        LOG.info(
            f"Contour grids already generated at {output_path}. Skipping generation."
        )
        return
    data_dict = {}
    data_dir = download_top1000_lib()
    if data_dir is None or not data_dir.exists():
        raise RuntimeError(
            "Top8000 rotamer library not found; cannot generate contour grids."
        )
    all_grid_files = list(data_dir.glob(f"{grid_dirname}/{grid_data_tag}-*.data"))

    if files_to_skip is not None:
        all_grid_files = [f for f in all_grid_files if f.name not in files_to_skip]

    if not all_grid_files:
        raise RuntimeError(f"No contour grid files found in {data_dir / grid_dirname}")
    LOG.info(
        f"Found {len(all_grid_files)} contour grid files in {data_dir / grid_dirname}"
    )
    for grid_file in all_grid_files:
        resname_tag = grid_file.stem.split("-")[
            1
        ].lower()  # e.g. "rota8000-arg.data" -> "arg"

        LOG.info(f"Generating contour grid for {grid_file.stem}...")
        tmp_contour = load_contour_grid_text(grid_file)
        if tmp_contour is None:
            LOG.warning(f"No contour grid found for {grid_file.stem}, skipping.")
            continue
        data_dict[resname_tag] = tmp_contour

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)
    LOG.info("All contour grids generated and saved.")


@functools.lru_cache
def load_all_contour_grid_from_pickle(
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
        LOG.info(
            f"Contour grids pickle file {contour_path} does not exist. Generating..."
        )
        generate_contour_grids_data(
            grid_dirname=grid_dirname,
            grid_data_tag=grid_data_tag,
            files_to_skip=files_to_skip,
        )
    with open(contour_path, "rb") as f:
        data = pickle.load(f)
    return data


def interp_wrapped(
    grid_obj: dict[str, np.ndarray],
    coords_deg: list[float],
) -> tuple[float, list[float]]:
    """
    Interpolate value at given χ coords (deg), handling wrapping axes.

    Parameters
    ----------
    grid_obj : dict[str, np.ndarray]
        The grid object containing 'grid', 'axes', and 'wrap' information.
    coords_deg : list[float]
        List of coordinates in degrees, which could be [phi, psi] / [χ1, χ2, ...].

    Returns
    -------
    tuple[float, list[float]]
        A tuple containing:
        - The interpolated value at the given coordinates.
        - The wrapped coordinates used for interpolation.
    """
    axes = []
    coords = []
    for val, centers, span, wrap in zip(
        coords_deg, grid_obj["axes"], grid_obj["span"], grid_obj["wrap"]
    ):
        if wrap:
            if (span[0] == 0) and (span[1] == 360):
                # Wrap to [0, 360)
                val = wrap_angle(val, mode="0-360")
            elif (span[0] == -180) and (span[1] == 180):
                # Wrap to [-180, 180)
                val = wrap_angle(val, mode="±180")
            elif (span[0] == 0) and (span[1] == 180):
                # Wrap to [0, 180]
                val = wrap_angle(val, mode="0-180")
        coords.append(val)
        axes.append(centers)
    interp_func = RegularGridInterpolator(
        tuple(axes), grid_obj["grid"], bounds_error=False, fill_value=np.nan
    )
    return float(interp_func(coords)[0]), coords


def check_rotamer(
    atom_array: struc.AtomArray, res_id: int, chain_id: str, model_no: int
) -> dict[str, Any]:
    """
    Check the rotamer classification for a given residue in the atom array.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The AtomArray containing the structure.
    res_id : int
        The residue ID to check.
    chain_id : str
        The chain ID to check.
    model_no : int
        The model number to check (usually 0 for single model structures).

    Returns
    -------
    dict[str, Any]
        A dictionary containing the rotamer classification results:
        {
            "model_no": int,
            "resname": str,
            "resid": int,
            "observed": dict,
            "chain_id": str,
            "rotamer_score_pct": float,
            "classification": str,
            "error": str | None
        }.
    """
    if not isinstance(atom_array, struc.AtomArray):
        raise TypeError("atom_array must be a biotite structure AtomArray")
    mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id)
    if not np.any(mask):
        raise ValueError("Residue not found")

    resname = atom_array.res_name[mask][0].upper()
    observed = get_residue_chis(atom_array, mask)
    if not observed:
        return {
            "model_no": model_no,
            "resname": resname,
            "resid": int(res_id),
            "chain_id": chain_id,
            "observed": observed,
            "rotamer_score_pct": 1.0,  # ALA and GLY have no chi angles
            "classification": "UNKNOWN",
            "error": "no chis computable",
        }

    grid_obj = load_contour_grid_for_residue(
        grid_dirname="Top8000_rotamer_pct_contour_grids",
        resname_tag=grid_residue_map.get(resname),
        grid_data_tag="rota8000",
        files_to_skip=("rota8000-leu-raw.data",),  # Use rota8000-leu.data instead
    )

    # Build coords in same order as grid dims
    coords_list = []
    for i in range(len(grid_obj["axes"])):
        coords_list.append(observed.get(f"chi{i + 1}", 0.0))

    pct, wrapped_angles = interp_wrapped(grid_obj, coords_list)
    observed = {f"chi{i + 1}": wrapped_angles[i] for i in range(len(wrapped_angles))}

    if np.isnan(pct):
        classification = "UNKNOWN"

    elif pct >= ROTA_ALLOWED_THRESHOLD:
        classification = "FAVORED"
    elif pct >= ROTA_OUTLIER_THRESHOLD:
        classification = "ALLOWED"
    else:
        classification = "OUTLIER"

    return {
        "model_no": model_no,
        "resname": resname,
        "resid": int(res_id),
        "observed": observed,
        "chain_id": chain_id,
        "rotamer_score_pct": pct,
        "classification": classification,
        "error": None,
    }


def get_residue_phi_psi_omega(atom_array: struc.AtomArray) -> dict:
    """
    Get the phi, psi, and omega dihedral angles for a given residue.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The AtomArray containing the structure.

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
    phi = np.rad2deg(phi)
    psi = np.rad2deg(psi)
    omega = np.rad2deg(omega)

    # Remove invalid values (NaN) at first and last position
    phi = phi[1:-1]
    psi = psi[1:-1]
    omega = omega[1:-1]
    return {"phi": phi, "psi": psi, "omega": omega}


def assign_rama_types(atom_array: struc.AtomArray) -> dict:
    """
    Assign Ramachandran types to residues based on their phi, psi, and omega angles.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The AtomArray containing the structure.

    Returns
    -------
    dict
        A dictionary containing the phi, psi, omega angles and their corresponding Ramachandran types.
    """
    # Get phi, psi, omega angles
    # Note: This function assumes the atom_array is already cleaned and contains only relevant residues.
    # It should not contain water or other non-protein residues.
    if not isinstance(atom_array, struc.AtomArray):
        raise TypeError("atom_array must be a biotite structure AtomArray")

    phi_psi_omega_dict = get_residue_phi_psi_omega(atom_array)
    omega = phi_psi_omega_dict["omega"]

    _, res_names = struc.get_residues(atom_array)
    next_res_names = np.roll(res_names, -1)[
        1:-1
    ]  # Shifted by -1, remove first and last

    def rama_type(res_name: str, next_res_name: str, omega_val: float) -> str:
        if res_name == "GLY":
            return "gly"
        elif res_name == "PRO":
            return "cispro" if abs(omega_val) < 30 else "transpro"
        elif next_res_name == "PRO":
            return "prepro"
        elif res_name in ["ILE", "VAL"]:
            return "ileval"
        else:
            return "general"

    rama_types = [
        rama_type(res_name, next_res_name, omega_val)
        for res_name, next_res_name, omega_val in zip(
            res_names[1:-1], next_res_names, omega
        )
    ]
    phi_psi_omega_dict["rama_types"] = rama_types
    return phi_psi_omega_dict


def check_rama(
    phi: float,
    psi: float,
    omega: float,
    res_id: int,
    resname: str,
    chain_id: str,
    resname_tag: str,
    model_no: int = 0,
) -> dict[str, Any]:
    """
    Check the Ramachandran classification for a given residue based on its phi, psi, and omega angles.

    Parameters
    ----------
    phi : float
        The phi angle in degrees.
    psi : float
        The psi angle in degrees.
    omega : float
        The omega angle in degrees.
    res_id : int
        The residue ID to check.
    resname : str
        The residue name to check.
    chain_id : str
        The chain ID to check.
    resname_tag : str
        The residue name tag to use for classification (e.g. "general", "cispro", "gly").
    model_no : int, optional
        The model number to use for classification (default is 0).

    Returns
    -------
    dict[str, Any]
        A dictionary containing the Ramachandran classification result.
        {
        'resname': str,
        'resid': int,
        'observed': {'phi': float, 'psi': float, 'omega': float},
        'rotamer_score_pct': float,
        'classification': str
        'model_no': int,
        'chain_id': str
        }.

    Notes
    -----
    This function uses the Top8000 Ramachandran contour grids to classify the residue.
    It checks the phi, psi, and omega angles against the grids and assigns a classification
    of "FAVORED", "ALLOWED", or "OUTLIER" based on the percentage of favored conformations.
    If the residue is GLY or PRO, it uses special handling for those residues.
    The thresholds for classification are defined as follows:
    - FAVORED: pct >= RAMA_FAVORED_THRESHOLD
    - ALLOWED: pct >= RAMA_ALLOWED_THRESHOLD (or RAMA_GENERAL_ALLOWED_THRESHOLD for "general" residues,
    or RAMA_CISPRO_ALLOWED_THRESHOLD for "cispro" residues)
    - OUTLIER: pct < RAMA_ALLOWED_THRESHOLD (or RAMA_GENERAL_ALLOWED_THRESHOLD for "general" residues,
    or RAMA_CISPRO_ALLOWED_THRESHOLD for "cispro" residues)
    """

    grid_obj = load_contour_grid_for_residue(
        grid_dirname="Top8000_ramachandran_pct_contour_grids",
        resname_tag=resname_tag,
        grid_data_tag="rama8000",
        files_to_skip=None,
    )

    # Build coords in same order as grid dims
    coords_list = [phi, psi]
    pct, wrapped_coords = interp_wrapped(grid_obj, coords_list)
    if resname_tag == "general":
        if pct >= RAMA_GENERAL_ALLOWED_THRESHOLD:
            classification = "ALLOWED"
        else:
            classification = "OUTLIER"
    elif resname_tag == "cispro":
        if pct >= RAMA_CISPRO_ALLOWED_THRESHOLD:
            classification = "ALLOWED"
        else:
            classification = "OUTLIER"
    else:
        if pct >= RAMA_ALLOWED_THRESHOLD:
            classification = "ALLOWED"
        else:
            classification = "OUTLIER"
    if pct >= RAMA_FAVORED_THRESHOLD:
        classification = "FAVORED"

    return {
        "model_no": model_no,
        "chain_id": chain_id,
        "resname": resname,
        "resid": int(res_id),
        "observed": {
            "phi": wrapped_coords[0],
            "psi": wrapped_coords[1],
            "omega": omega,
        },
        "rama_score_pct": pct,
        "classification": classification,
    }


class ResidueRotamerScore(BaseModel):
    """
    Rotamer score for a residue in a protein structure.

    This class represents the rotamer score for a single residue, including its name,
    ID, observed chi angles, rotamer score percentage, and classification.

    Attributes
    ----------
    model_no : int
        The model number for the residue.
    resname : str
        The name of the residue (e.g. "ARG", "LEU").
    resid : int
        The ID of the residue in the structure.
    chain_id : str
        The chain ID of the residue.
    observed : dict[str, float]
        A dictionary containing the observed chi angles for the residue, e.g. {'chi1': 60.0, 'chi2': -45.0}.
    rotamer_score_pct : float
        The rotamer score percentage for the residue, e.g. 0.01 for 1% favored.
    classification : str
        The classification of the rotamer score, which can be one of the following:
        - "FAVORED": The residue is in a favored rotamer conformation.
        - "ALLOWED": The residue is in an allowed rotamer conformation.
        - "OUTLIER": The residue is in an outlier rotamer conformation.
        - "UNKNOWN": The residue's rotamer score could not be determined.
    error : str | None
        An error message if any error occurred during rotamer scoring, otherwise None.
    """

    model_config = ConfigDict(from_attributes=True)
    model_no: int = Field(
        description="Model number for the residue, default is 0", default=0
    )
    resname: str = Field(description="Residue name")
    resid: int = Field(description="Residue ID")
    chain_id: str = Field(description="Chain ID of the residue")
    observed: dict[str, float] = Field(
        description="Observed chi angles for the residue, e.g. {'chi1': 60.0, 'chi2': 170.0}"
    )
    rotamer_score_pct: float = Field(
        description="Rotamer score percentage, e.g. 0.01 for 1% favored"
    )
    classification: str = Field(
        description="Rotamer classification: FAVORED, ALLOWED, OUTLIER, UNKNOWN",
        examples=["FAVORED", "ALLOWED", "OUTLIER", "UNKNOWN"],
    )
    error: str | None = Field(
        description="Error message if any error occurred during rotamer scoring",
        default=None,
    )

    @classmethod
    def from_residue(
        cls, atom_array: struc.AtomArray, res_id: int, chain_id: str, model_no: int
    ) -> "ResidueRotamerScore":
        """
        Create ResidueRotamerScore from a residue object.

        Parameters
        ----------
        atom_array : struc.AtomArray
            The AtomArray containing the structure.
        res_id : int
            The residue ID to check.
        chain_id : str
            The chain ID to check.
        model_no : int
            The model number to check (usually 0 for single model structures).

        Returns
        -------
        ResidueRotamerScore
            An instance of ResidueRotamerScore containing the rotamer score for the residue.
        """
        result = check_rotamer(atom_array, res_id, chain_id, model_no=model_no)
        return cls(**result)


class RotamerScore(BaseModel):
    """
    Rotamer score for a given protein structure.
    This class represents the rotamer scores for all residues in a protein structure,
    including their names, IDs, observed chi angles, rotamer score percentages, and classifications.
    """

    model_config = ConfigDict(from_attributes=True)
    rotamer_scores: list[ResidueRotamerScore] | list[list[ResidueRotamerScore]] = Field(
        description="List of rotamer scores for each residue in the structure."
    )

    @classmethod
    def from_atom_array(
        cls, atom_array: struc.AtomArray, model_no: int = 0
    ) -> "RotamerScore":
        """
        Create ResidueRotamerScore from a protein structure.

        Parameters
        ----------
        atom_array : struc.AtomArray
            The AtomArray containing the structure.
        model_no : int, optional
            The model number to use for classification (default is 0).

        Returns
        -------
        RotamerScore
            An instance of RotamerScore containing the rotamer scores for the structure.
        """
        if not isinstance(atom_array, struc.AtomArray):
            LOG.warning(
                f"RotamerScore: Cannot be computed for model {model_no}; atom_array must be a biotite structure AtomArray"
            )
            return cls(rotamer_scores=[])

        if all(atom_array.hetero):
            LOG.warning(
                f"RotamerScore: Cannot be computed for model {model_no}; atom_array must contain some protein residues"
            )
            return cls(rotamer_scores=[])
        atom_array = atom_array[
            struc.filter_amino_acids(atom_array) & ~atom_array.hetero
        ]
        if atom_array.array_length == 0:
            LOG.warning(
                f"RotamerScore: Cannot be computed for model {model_no};atom_array must contain at least one residue"
            )
            return cls(rotamer_scores=[])
        atom_array = atom_array[~atom_array.hetero]  # Remove hetero
        res_ids, res_names = struc.get_residues(atom_array)
        rotamer_scores = []
        for res_id, res_name in zip(res_ids, res_names):
            chain_id = atom_array.chain_id[atom_array.res_id == res_id][0]
            if (
                res_name not in grid_residue_map
                or res_name == "GLY"
                or res_name == "ALA"
            ):
                continue
            rotamer_scores.append(
                ResidueRotamerScore.from_residue(
                    atom_array=atom_array,
                    res_id=res_id,
                    chain_id=chain_id,
                    model_no=model_no,
                )
            )

        return cls(rotamer_scores=rotamer_scores)

    @classmethod
    def from_atom_array_or_stack(
        cls, atom_array: struc.AtomArray | struc.AtomArrayStack
    ) -> "RotamerScore":
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
        if isinstance(atom_array, struc.AtomArray):
            return cls.from_atom_array(atom_array, model_no=0)
        elif isinstance(atom_array, struc.AtomArrayStack):
            rotamer_scores = []
            for model_no, atom_arr in enumerate(atom_array):
                rotamer_scores.append(
                    cls.from_atom_array(atom_arr, model_no=model_no).rotamer_scores
                )
            return cls(rotamer_scores=rotamer_scores)
        else:
            raise TypeError(
                "atom_array must be a biotite structure AtomArray or AtomArrayStack"
            )


class ResidueRamaScore(BaseModel):
    """
    Ramachandran score for a residue in a protein structure.
    This class represents the ramachandran score for a single residue, including its name,
    ID, observed phi, psi, omega angles, Ramachandran score percentage, and classification.
    """

    resname: str = Field(description="Residue name")
    resid: int = Field(description="Residue ID")
    chain_id: str = Field(description="Chain ID of the residue")
    resname_tag: str = Field(
        description="Residue name tag for classification, e.g. 'general', 'cispro', 'gly'"
    )
    model_no: int = Field(
        description="Model number for the residue, default is 0", default=0
    )
    observed: dict[str, float] = Field(
        description="Observed phi, psi, omega angles for the residue, e.g. {'phi': -60.0, 'psi': 45.0, 'omega': 180.0}"
    )
    rama_score_pct: float = Field(
        description="Ramachandran score percentage, e.g. 0.01 for 1% favored"
    )
    classification: str = Field(
        description="Ramachandran classification: FAVORED, ALLOWED, OUTLIER",
        examples=["FAVORED", "ALLOWED", "OUTLIER"],
    )

    @classmethod
    def from_phi_psi_omega(
        cls,
        phi: float,
        psi: float,
        omega: float,
        res_id: int,
        resname: str,
        chain_id: str,
        resname_tag: str,
        model_no: int,
    ) -> "ResidueRamaScore":
        result = check_rama(
            phi, psi, omega, res_id, resname, chain_id, resname_tag, model_no
        )
        result["resname_tag"] = resname_tag
        return cls(**result)


class RamaScore(BaseModel):
    """
    Ramachandran score for a given protein structure.
    This class represents the ramachandran scores for all residues in a protein structure,
    including their names, IDs, observed phi, psi, omega angles, Ramachandran score percentages, and classifications.
    It is used to assess the quality of the protein structure based on the ramachandran plot
    and to identify potential outliers in the ramachandran angles.
    """

    model_config = ConfigDict(from_attributes=True)
    rama_scores: list[ResidueRamaScore] | list[list[ResidueRamaScore]] = Field(
        description="List of ramachandran scores for each residue in the structure."
    )

    @classmethod
    def from_atom_array(
        cls, atom_array: struc.AtomArray, model_no: int = 0
    ) -> "RamaScore":
        """
        Create RamaScore from a protein structure.

        Parameters
        ----------
        atom_array : struc.AtomArray
            The AtomArray containing the structure.
        model_no : int, optional
            The model number to use for classification (default is 0).

        Returns
        -------
        RamaScore
            An instance of RamaScore containing the ramachandran scores for the structure.
        """
        if all(atom_array.hetero):
            LOG.warning(
                f"RotamerScore: Cannot be computed for model {model_no}; atom_array must contain some protein residues"
            )
            return cls(rama_scores=[])
        if not isinstance(atom_array, struc.AtomArray):
            LOG.warning(
                f"RamaScore: Cannot be computed for model {model_no}; atom_array must be a biotite structure AtomArray"
            )
            return cls(rama_scores=[])
        atom_array = atom_array[
            struc.filter_amino_acids(atom_array) & ~atom_array.hetero
        ]
        if atom_array.array_length == 0:
            LOG.warning(
                f"RamaScore: Cannot be computed for model {model_no};atom_array must contain at least one residue"
            )
            return cls(rama_scores=[])

        rama_input = assign_rama_types(atom_array)
        res_ids, res_names = struc.get_residues(atom_array)
        rama_scores = []
        for phi, psi, omega, res_id, res_name, rama_type in zip(
            rama_input["phi"],
            rama_input["psi"],
            rama_input["omega"],
            res_ids[1:-1],
            res_names[1:-1],
            rama_input["rama_types"],
        ):
            rama_scores.append(
                ResidueRamaScore.from_phi_psi_omega(
                    phi=phi,
                    psi=psi,
                    omega=omega,
                    res_id=res_id,
                    resname=res_name,
                    chain_id=atom_array.chain_id[atom_array.res_id == res_id][0],
                    resname_tag=rama_type,
                    model_no=model_no,
                )
            )

        return cls(rama_scores=rama_scores)

    @classmethod
    def from_atom_array_or_stack(
        cls, atom_array: struc.AtomArray | struc.AtomArrayStack
    ) -> "RamaScore":
        """
        Create RamaScore from a protein structure or a stack of structures.

        Parameters
        ----------
        atom_array : struc.AtomArray | struc.AtomArrayStack
            The AtomArray or AtomArrayStack containing the structure(s).

        Returns
        -------
        RamaScore
            An instance of RamaScore containing the ramachandran scores for the structure(s).
        """
        if isinstance(atom_array, struc.AtomArray):
            return cls.from_atom_array(atom_array, model_no=0)
        elif isinstance(atom_array, struc.AtomArrayStack):
            rama_scores = []
            for model_no, atom_arr in enumerate(atom_array):
                rama_scores.append(
                    cls.from_atom_array(atom_arr, model_no=model_no).rama_scores
                )
            return cls(rama_scores=rama_scores)
        else:
            raise TypeError(
                "atom_array must be a biotite structure AtomArray or AtomArrayStack"
            )


def get_fraction_of_rotamer_outliers(atom_array: struc.AtomArray) -> float:
    """
    Compute the fraction of rotamer outliers for given model.

    Parameters
    ----------
    atom_array : struc.AtomArray
        The AtomArray containing the structure.

    Returns
    -------
    float
        The fraction of rotamer outliers in the structure, calculated as the number of outlier
        rotamers divided by the total number of rotamers.
    """
    outlier_rotamers = 0
    total_rotamers = 0
    result = RotamerScore.from_atom_array_or_stack(atom_array)
    for rotamer_score in result.rotamer_scores:
        if isinstance(rotamer_score, list):
            # If it's a stack, we need to iterate through each score
            for score in rotamer_score:
                if score.classification == "OUTLIER":
                    outlier_rotamers += 1
                total_rotamers += 1
        elif isinstance(rotamer_score, ResidueRotamerScore):
            # If it's a single score, check its classification
            if rotamer_score.classification == "OUTLIER":
                outlier_rotamers += 1
            total_rotamers += 1
    return outlier_rotamers / total_rotamers if total_rotamers > 0 else 0.0


def get_fraction_of_rama_outliers(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
) -> float:
    """
    Compute the fraction of ramachandran outliers for given model.

    Parameters
    ----------
    atom_array : struc.AtomArray | struc.AtomArrayStack
        The AtomArray or AtomArrayStack containing the structure(s).

    Returns
    -------
    float
        The fraction of ramachandran outliers in the structure, calculated as the number of outlier
        ramachandran angles divided by the total number of ramachandran angles.
    """
    outlier_rama = 0
    total_rama = 0
    result = RamaScore.from_atom_array_or_stack(atom_array)

    for rama_score in result.rama_scores:
        if isinstance(rama_score, list):
            # If it's a stack, we need to iterate through each score
            for score in rama_score:
                if score.classification == "OUTLIER":
                    outlier_rama += 1
                total_rama += 1
        elif isinstance(rama_score, ResidueRamaScore):
            if rama_score.classification == "OUTLIER":
                outlier_rama += 1
            total_rama += 1
    return outlier_rama / total_rama if total_rama > 0 else 0.0


if __name__ == "__main__":
    from biotite.structure import io as strucio

    cif_path = Path("/Users/yusuf/peppr/test_small_mol.cif")
    atom_array = strucio.load_structure(cif_path)
    rotamer_score = RotamerScore.from_atom_array(atom_array)
    rama_score = RamaScore.from_atom_array(atom_array)
    print(rotamer_score)
    print(rama_score)
