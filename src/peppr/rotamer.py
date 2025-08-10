"""
Backbone-dependent and backbone-independent rotamer checking using
Richardson Lab Top8000 rotamer library.
https://pmc.ncbi.nlm.nih.gov/articles/PMC4983197/

This is the library useful programs like MolProbity's "rotamer" validation, which uses
mmtbx.rotamer.rotamer_eval and mmtbx.validation.rotalyze under the hood.
"""
import functools
import logging
import math
import requests
import numpy as np
from pathlib import Path
import biotite.structure as struc

from biotite.structure.io import load_structure
from biotite import structure as struc
import pickle
from scipy.interpolate import RegularGridInterpolator

# Configure basic logging to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

OUTLIER_THRESHOLD = 0.003
ALLOWED_THRESHOLD = 0.02


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
    "VAL": "val"
}

# ---------- URLs (raw GitHub) ----------
# NOTE: the exact path in the repo may vary; change paths if needed
ROTAMER_CSV_URL = ("https://raw.githubusercontent.com/rlabduke/reference_data/refs/heads/master/Top8000/Top8000_rotamer_central_values")

# The contour grids are stored per-residue in the repo under:
# Top8000/Top8000_rotamer_pct_contour_grids/
# (these are numpy/text arrays; file names vary by residue)
CONTOUR_BASE_URL = ("https://raw.githubusercontent.com/rlabduke/reference_data/master/"
                    "Top8000/Top8000_rotamer_pct_contour_grids/")

_ROTAMER_BASE_DIR =  Path.home() / ".local/share/rotamer_lib"
_ROTAMER_DIR = _ROTAMER_BASE_DIR / "reference_data-master" / "Top8000"


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
        The destination path where the file should be saved. If None, uses the URL's name

    Returns
    -------
    Path | None
        The path where the file was saved or extracted.
    """
    if dest_path is None:
        dest_path = Path(url).name
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
        if str(dest_path).endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(dest_path.parent)
            return dest_path.parent
        elif str(dest_path).endswith('.tar.gz') or str(dest_path).endswith('.tgz'):
            import tarfile
            with tarfile.open(dest_path, 'r:gz') as tar_ref:
                tar_ref.extractall(dest_path.parent)
            return dest_path.parent
    return None

# ---------- Biotite helpers: compute chi dihedrals ----------
def get_residue_chis(atom_array: struc.AtomArray, res_id_mask: np.ndarray) -> dict:
    """
    Use Biotite helpers to compute chi angles for a residue.
    - atom_array: AtomArray
    - res_id_mask: boolean mask selecting atoms of the residue (or an index for residue)
    Returns dict like {'chi1': angle_deg, 'chi2': ...}
    """
    # Biotite has a convenience: structure's sidechain_dihedrals or similar don't exist as single helper.
    # We'll compute common chis by looking up named atoms for each residue type.
    res_atoms = atom_array[res_id_mask]
    # Build a map name -> coordinates
    coords = {name.strip(): coord for name, coord in zip(res_atoms.atom_name, res_atoms.coord)}
    # mapping of chi definitions per residue: set of atom name tuples (A,B,C,D) for chi1..chi4
    # This is a small mapping; for production you'd want full canonical table (omitted for brevity).
    CHI_DEFS = {
        # Example: for ARG: chi1: N-CA-CB-CG  (names: 'N','CA','CB','CG' )
        # Many residues must be filled in; for the demo we include a few.
        'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
        'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],  # LEU uses chi2 to CD1 or CD2 depending; this is simplified
        'VAL': [('N','CA','CB','CG1')],
        'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
        'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
        'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
        'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1'), ('CB','CG','CD1','CE1')],
        'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1'), ('CB','CG','CD1','CE2')],
        'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1'), ('CB','CG','CD1','CE1')],
        'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
        'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','OE1')],
        'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
        'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','OE1')],
        'CYS': [('N','CA','CB','SG')],
        'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
        'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD')],
        'SER': [('N','CA','CB','OG')],
        'THR': [('N','CA','CB','OG1')],
        'ALA': [],  # ALA has no chi angles
        'GLY': [],  # GLY has no chi angles
    }
    # detect residue type name
    resname = res_atoms.res_name[0].upper()
    chis = {}
    if resname not in CHI_DEFS:
        return chis
    from biotite.structure import dihedral
    for i, atom_tuple in enumerate(CHI_DEFS[resname], start=1):
        try:
            p = [coords[a] for a in atom_tuple]
            ang = math.degrees(dihedral(*p))
            chis[f"chi{i}"] = ang
        except Exception as e:
            # Skip if any atom is missing
            pass
    return chis


@functools.lru_cache
def download_top1000_rotamer_lib() -> Path:
    """
    Download the Top8000 rotamer library from the Richardson Lab GitHub.
    This is a convenience function to download the entire library.
    """
    if _ROTAMER_DIR.exists():
        return _ROTAMER_DIR
    url = "https://github.com/rlabduke/reference_data/archive/refs/heads/master.zip"
    data_dir = download_and_extract(url, _ROTAMER_BASE_DIR/ "top1000.zip")
    if data_dir is None:
        raise RuntimeError("Failed to download or extract Top8000 rotamer library")
    LOG.info(f"Top8000 rotamer library downloaded and extracted to {data_dir}")
    return _ROTAMER_DIR


@functools.lru_cache
def load_contour_grid_text(resname_tag: str) -> dict[str, np.ndarray]:
    """
    Parse a Top8000 pct_contour_grid text file into a numpy grid and axis coordinate arrays.
    Returns:
      {
        'grid': np.ndarray,
        'axes': [axis1_centers, axis2_centers, ...],
        'wrap': [True/False per axis]
      }
    """
    # Download data
    grid_data = _ROTAMER_DIR / f"Top8000_rotamer_pct_contour_grids/rota8000-{resname_tag}.data"
    if isinstance(grid_data, Path):
        with open(grid_data, "r") as f:
            text = f.read()
    else:
        text = grid_data
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Parse header
    n_dims = 0
    axes_meta = []
    data_start_idx = 0
    for i, ln in enumerate(lines):
        if ln.lower().startswith("# number of dimensions"):
            n_dims = int(ln.split(":")[1])
        elif ln.strip().startswith("#   x"):
            # Format: "#   x1: 0.0  360.0  36 true"
            parts = ln.split(":")[1].split()
            low, high, nbins, wrap = float(parts[0]), float(parts[1]), int(parts[2]), parts[3].lower() == "true"
            axes_meta.append((low, high, nbins, wrap))
        elif not ln.startswith("#"):
            data_start_idx = i
            break
    # Build coordinate arrays (bin centers)
    axes_coords = []
    wraps = []
    for (low, high, nbins, wrap) in axes_meta:
        step = (high - low) / nbins
        centers = np.linspace(low + step/2, high - step/2, nbins)
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
            bin_idx = int((coords[dim] - low) / step) % nbins if wrap else int((coords[dim] - low) / step)
            idxs.append(bin_idx)
        grid[tuple(idxs)] = val

    return {
        'grid': grid,
        'axes': axes_coords,
        'wrap': wraps
    }

@functools.lru_cache
def get_contour_grid(resname: str) -> dict[str, np.ndarray]:
    resname_tag = grid_residue_map.get(resname)
    if resname_tag is None:
        return
    return load_contour_grid_text(resname_tag)

@functools.lru_cache
def load_contour_grid_for_residue(resname: str) -> dict[str, np.ndarray]:
    """
    Load contour grid for a given residue name.
    Returns a dict with 'grid', 'axes' (list of axis arrays), and 'wrap' (list of bools).
    """
    resname_tag = grid_residue_map.get(resname.upper())
    if resname_tag is None:
        raise ValueError(f"No contour grid mapping for residue {resname}")

    # Check if the grid is already loaded
    all_dict_contour = load_all_contour_grid_from_pickle()
    if resname in all_dict_contour:
        return all_dict_contour[resname]


@functools.lru_cache
def generate_contour_grids_data() -> None:
    """
    """
    # Save dictionary to a single pickle file
    output_path = _ROTAMER_DIR / "Top8000_rotamer_pct_contour_grids.pkl"
    if output_path.exists():
        LOG.info(f"Contour grids already generated at {output_path}. Skipping generation.")
        return
    data_dict = {}
    data_dir = download_top1000_rotamer_lib()
    if data_dir is None or not data_dir.exists():
        raise RuntimeError("Top8000 rotamer library not found; cannot generate contour grids.")
    for resname in grid_residue_map.keys():
        LOG.info(f"Generating contour grid for {resname}...")
        tmp_contour = get_contour_grid(resname)
        if tmp_contour is None:
            LOG.warning(f"No contour grid found for {resname}, skipping.")
            continue
        data_dict[resname] = tmp_contour

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)
    LOG.info("All contour grids generated and saved.")


@functools.lru_cache
def load_all_contour_grid_from_pickle() -> dict[str, dict[str, np.ndarray]]:
    """
    Load all contour grids for a given residue name from a pickle file.
    """
     # Save dictionary to a single pickle file
    contour_path = _ROTAMER_DIR / "Top8000_rotamer_pct_contour_grids.pkl"
    if not contour_path.exists():
        LOG.info(f"Contour grids pickle file {contour_path} does not exist. Generating...")
        generate_contour_grids_data()
    with open(contour_path, "rb") as f:
        data = pickle.load(f)
    return data


def interp_wrapped(grid_obj, coords_deg):
    """Interpolate value at given χ coords (deg), handling wrapping axes."""
    axes = []
    coords = []
    for val, centers, wrap in zip(coords_deg, grid_obj['axes'], grid_obj['wrap']):
        if wrap:
            val = val % 360.0
        coords.append(val)
        axes.append(centers)
    interp_func = RegularGridInterpolator(tuple(axes), grid_obj['grid'], bounds_error=False, fill_value=np.nan)
    return float(interp_func(coords))


def check_rotamer_backbone_dependent(atom_array, res_id, chain_id):
    """
    Compute rotamer classification based on backbone-dependent pct_contour_grids.
    Uses grid text parser for wrapping and bin centers.
    """

    mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id)
    if not np.any(mask):
        raise ValueError("Residue not found")

    resname = atom_array.res_name[mask][0].upper()
    observed = get_residue_chis(atom_array, mask)
    if not observed:
        return {'resname': resname, 'error': 'no chis computable'}

    grid_obj = load_contour_grid_for_residue(resname)

    # Build coords in same order as grid dims
    coords_list = []
    for i in range(len(grid_obj['axes'])):
        coords_list.append(observed.get(f"chi{i+1}", 0.0))

    pct = interp_wrapped(grid_obj, coords_list)

    if np.isnan(pct):
        classification = "UNKNOWN"

    elif pct >= ALLOWED_THRESHOLD:
        classification = "FAVORED"
    elif pct >= OUTLIER_THRESHOLD:
        classification = "ALLOWED"
    else:
        classification = "OUTLIER"

    return {
        'resname': resname,
        'observed': observed,
        'rotamer_score_pct': pct,
        'classification': classification
    }


def get_fraction_of_rotamer_outliers(atom_array: struc.AtomArray) -> float:
    """
    Compute the fraction of rotamer outliers for given model.
    """
    outlier_rotamers = 0
    total_rotamers = 0
    res_ids, res_names = struc.get_residues(atom_array)
    for res_id, res_name in zip(res_ids, res_names):
        chain_id = atom_array.chain_id[atom_array.res_id == res_id][0]
        # Check rotamer classification
        if res_name not in grid_residue_map or res_name == "GLY" or res_name == "ALA":
            continue
        result = check_rotamer_backbone_dependent(atom_array, res_id, chain_id)
        if result.get('classification') == "OUTLIER":
            # If any residue is an outlier, return 1.0
            outlier_rotamers += 1
        total_rotamers += 1
    return outlier_rotamers / total_rotamers if total_rotamers > 0 else 0.0


if __name__ == "__main__":
    # Example usage of loading contour grids for all residues
    #generate_contour_grids_npz_data()
    #atom_array = atom_array_from_file("/Users/yusuf/1yhr.pdb")  # change to your file
    atom_array = load_structure("/Users/yusuf/1ohj.pdb")  # change to your file
    a=check_rotamer_backbone_dependent(atom_array, 122, "A")



