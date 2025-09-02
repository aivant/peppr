from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, ConvexHull, KDTree
from scipy.spatial.qhull import QhullError
from scipy.constants import R

import biotite.structure as struc
from biotite.structure.info import vdw_radius_single


@dataclass(frozen=True)
class ResidueKey:
    """Key for residue identification in a structure.

    Attributes
    ----------
    chain_id : str
        The chain identifier for the residue.
    res_id : int
        The residue identifier (sequence number) for the residue.
    """
    chain_id: str
    res_id: int


@dataclass
class ResiduePacking:
    """
    Result of packing entropy calculation for a single residue.

    Attributes
    ----------
    chain_id : str
        The chain identifier for the residue.
    res_id : int
        The residue identifier (sequence number) for the residue.
    res_name : str
        The residue name (three-letter code).
    packing_fraction : float
        The packing fraction of the residue.
    packing_entropy : float
        The packing entropy of the residue.
    """
    chain_id: str
    res_id: int
    res_name: str
    packing_fraction: float
    packing_entropy: float


def _residue_spans(atoms: struc.AtomArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get residue start/stop indices for slicing into AtomArray by residue.

    Parameters
    ----------
    atoms : struc.AtomArray
        The AtomArray to analyze.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The start and stop indices for each residue.
    """
    starts = struc.get_residue_starts(atoms)
    stops = [*starts[1:], len(atoms)]
    return starts, stops


def _residue_keys(atoms: struc.AtomArray) -> List[ResidueKey]:
    """
    Get residue keys for all residues in the AtomArray.

    Parameters
    ----------
    atoms : struc.AtomArray
        The AtomArray to analyze.

    Returns
    -------
    List[ResidueKey]
        The list of residue keys for all residues in the AtomArray.
    """
    starts, stops = _residue_spans(atoms)
    keys: List[ResidueKey] = []
    for s, e in zip(starts, stops):
        # all atoms in residue share chain_id and res_id
        keys.append(ResidueKey(chain_id=str(atoms.chain_id[s]), res_id=int(atoms.res_id[s])))
    return keys


def _fibonacci_sphere(n: int) -> np.ndarray:
    """
    Points (unit sphere) using the Fibonacci method. Shape: (n, 3)
    Reference: https://stackoverflow.com/a/26127012
    """
    # indices = 0.5, 1.5, ... n-0.5
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2.0 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack((x, y, z), axis=1)


def _surface_points_for_atom(
    center: np.ndarray,
    element: str,
    kd: KDTree,
    probe: float,
    n_on_sphere: int,
) -> np.ndarray:
    """
    Create outward surface points around one atom and cull those that
    are closer than the sphere diameter to any other atom center.
    """
    # radius used in original: (probe + vdw_radius) * 2  (then used as both multiplier and cutoff)
    # We'll match that behavior for fidelity.
    radius = vdw_radius_single(element)
    if radius is None:
        radius = 1.7 # vdw for carbon
    sphere_diam = 2.0 * (probe + radius)

    # Unit sphere directions
    dirs = _fibonacci_sphere(n_on_sphere)  # (N,3)
    pts = center[None, :] + dirs * sphere_diam  # (N,3)

    # keep only points whose nearest neighbor (excluding self) is > sphere_diam
    # kd.query returns (dists, idx); with k=2, idx[0] is self for atom centers array; we used atom centers
    # in the KDTree, so take the second neighbor.
    dists, idxs = kd.query(pts, k=2)
    keep = dists[:, 1] > sphere_diam
    return pts[keep]


def _compute_surface_points(
    coords: np.ndarray,
    elements: np.ndarray,
    probe: float,
    n_on_sphere: int,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Generate surface points for all atoms.
    """
    kd = KDTree(coords)

    # Threaded map – light and simple
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(processes=n_threads) if n_threads else ThreadPool()
    try:
        tasks = [
            (coords[i], elements[i], kd, probe, n_on_sphere)
            for i in range(coords.shape[0])
        ]
        results = pool.starmap(_surface_points_for_atom, tasks)
    finally:
        pool.close()
        pool.join()

    if len(results) == 0:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(results, axis=0) if any(len(r) for r in results) else np.empty((0, 3))


def _convex_hull_volume_safe(points: np.ndarray) -> float:
    """
    Robust hull volume with graceful fallback for degeneracies/small sets.
    """
    # ConvexHull in 3D requires at least 4 non-coplanar points
    if points.shape[0] < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def compute_packing_entropy(
    atoms_in: Union[struc.AtomArray, struc.AtomArrayStack],
    chains: Optional[Union[str, Iterable[str]]] = None,
    probe_size: float = 1.4,
    onspherepoints: int = 30,
    n_threads: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute per-residue packing fraction and packing entropy for a protein
    using biotite structures + SciPy Voronoi/ConvexHull.

    Parameters
    ----------
    atoms_in
        biotite AtomArray or AtomArrayStack
    chains
        None = all chains; string for one chain; iterable[str] for multiple
    probe_size
        Solvent probe radius (Å). Use 1.4 Å for water (default).
    onspherepoints
        Points placed on the sphere per atom (Fibonacci sphere).
    n_threads
        Worker threads for generating surface points. Default: cpu-bound
        speed is typically fine because work is mostly KDTree queries.

    Returns
    -------
    DataFrame with columns:
        ['chain_id', 'res_id', 'res_name', 'packing_fraction', 'packing_entropy']
    """
    if chains is None:
        chains = list(set(atoms_in.chain_id))
    atoms = atoms_in[np.isin(atoms_in.chain_id, chains)]

    coords = atoms.coord.astype(float)  # (N,3)
    elements = atoms.element  # (N,)
    res_names = atoms.res_name
    chain_ids = atoms.chain_id
    res_ids = atoms.res_id

    # Build residue indexing
    starts, stops = _residue_spans(atoms)
    res_keys = _residue_keys(atoms)

    # 1) Generate surface points
    surface_pts = _compute_surface_points(
        coords=coords,
        elements=elements,
        probe=probe_size,
        n_on_sphere=onspherepoints,
        n_threads=n_threads,
    )

    # 2) Voronoi on protein points + surface points
    #    Keep in mind Voronoi may contain -1 (infinite) regions; skip those vertices.
    if surface_pts.shape[0] == 0:
        # Degenerate: no surface points accepted -> Voronoi would be pointless
        # We can still compute PF with just residue/atom hull volumes,
        # but it will likely be 1.0 (or undefined). Follow original idea and
        # return zeros safely.
        df = []
        for (s, e), key in zip(zip(starts, stops), res_keys):
            res_coords = coords[s:e]
            # PF undefined -> set to np.nan and entropy to np.nan to be explicit
            df.append(
                ResiduePacking(
                    chain_id=key.chain_id,
                    res_id=key.res_id,
                    res_name=str(res_names[s]),
                    packing_fraction=np.nan,
                    packing_entropy=np.nan,
                )
            )


        return pd.DataFrame(df)

    all_points = np.vstack((surface_pts, coords))
    vor = Voronoi(all_points)

    # 3) Collect "available volume" vertices per residue
    #    For each atom's region (shifted by len(surface_pts)), gather vertices.
    n_surface = surface_pts.shape[0]
    avail_vertices: Dict[int, List[np.ndarray]] = {ri: [] for ri in range(len(starts))}

    #print(avail_vertices)
    # Map atom index -> residue index
    atom_to_res_index = np.empty(coords.shape[0], dtype=int)
    for r_idx, (s, e) in enumerate(zip(starts, stops)):
        atom_to_res_index[s:e] = r_idx

    # Regions: vor.point_region[point_index] -> region id
    # Regions for our atoms start at index offset n_surface
    # Each region maps to vor.regions[region_id] -> list of vertex indices (may include -1)
    for atom_idx in range(coords.shape[0]):
        region_id = vor.point_region[n_surface + atom_idx]
        region_vertices = vor.regions[region_id]
        if region_vertices is None or len(region_vertices) == 0:
            continue
        res_idx = atom_to_res_index[atom_idx]
        for v_id in region_vertices:
            if v_id == -1:
                # open / unbounded face; skip
                continue
            v = vor.vertices[v_id]
            if np.all(np.isfinite(v)):
                avail_vertices[res_idx].append(v)

    # 4) Packing Fraction per residue:
    #    PF = volume(ConvexHull(atoms in residue)) / volume(ConvexHull(available vertices))
    #    Guard against degeneracy (zero or invalid volumes)
    results = []
    for r_idx, (s, e) in enumerate(zip(starts, stops)):
        res_coords = coords[s:e]
        atoms_hull_vol = _convex_hull_volume_safe(res_coords)

        av_list = avail_vertices[r_idx]
        if len(av_list) == 0:
            avail_vol = 0.0
        else:
            avail_vol = _convex_hull_volume_safe(np.asarray(av_list))

        if atoms_hull_vol <= 0.0 or avail_vol <= 0.0:
            pf = np.nan
            pe = np.nan
        else:
            pf = float(atoms_hull_vol / avail_vol)
            # New model in the source: PE = -R * log10(PF)
            # Units of R are J/(mol*K) -> entropy in same units.
            # If you want bits or nat units adjust the log accordingly.
            pe = -R * np.log10(pf)

        results.append(ResiduePacking(
                    chain_id=str(chain_ids[s]),
                    res_id=int(res_ids[s]),
                    res_name=str(res_names[s]),
                    packing_fraction=pf,
                    packing_entropy=pe,
                ))


    return pd.DataFrame(results)


def summarize_packing_entropy(df: pd.DataFrame) -> dict:
    """
    Convenience helpers to match the 'total' methods in the original API.
    NaNs are ignored in sums.
    """
    total_pf = float(np.nansum(df["packing_fraction"].values))
    total_entropy = float(np.nansum(df["packing_entropy"].values))
    return {
        "total_packing_fraction": total_pf,
        "total_entropy": total_entropy,
    }


def chain_entropy(df: pd.DataFrame, chain_id: str) -> float:
    """
    Sum of residue packing entropies for a specific chain.
    """
    mask = df["chain_id"] == chain_id
    return float(np.nansum(df.loc[mask, "packing_entropy"].values))


class PackingEntropy:
    """
    Compute and store per-residue packing entropy for a protein using biotite.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        Structure to analyze.
    chains : str or list[str], optional
        Restrict calculation to one or more chains (default: all).
    probe_size : float, default=1.4
        Solvent probe radius (Å).
    onspherepoints : int, default=30
        Points on the sphere per atom.
    n_threads : int, optional
        Number of worker threads for surface-point generation.
    """

    def __init__(
        self,
        atoms: Union[struc.AtomArray, struc.AtomArrayStack],
        chains: Optional[Union[str, Iterable[str]]] = None,
        probe_size: float = 1.4,
        onspherepoints: int = 30,
        n_threads: Optional[int] = None,
    ):
        self.atoms = atoms[struc.filter_canonical_amino_acids(atoms)]
        self.chains = chains
        self.probe_size = probe_size
        self.onspherepoints = onspherepoints
        self.n_threads = n_threads

        # Perform calculation once at init
        self._df = compute_packing_entropy(
            self.atoms,
            chains=self.chains,
            probe_size=self.probe_size,
            onspherepoints=self.onspherepoints,
            n_threads=self.n_threads,
        )

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the full per-residue DataFrame with packing fraction and entropy.
        """
        return self._df.copy()

    def get_total_packing_fraction(self) -> float:
        """
        Sum of residue packing fractions (ignores NaN).
        """
        return float(np.nansum(self._df["packing_fraction"].values))

    def get_total_entropy(self) -> float:
        """
        Sum of residue packing entropies (ignores NaN).
        """
        return float(np.nansum(self._df["packing_entropy"].values))

    def get_total_chain_entropy(self, chain: str) -> float:
        """
        Sum of residue entropies for a given chain ID.
        """
        mask = self._df["chain_id"] == chain
        return float(np.nansum(self._df.loc[mask, "packing_entropy"].values))

    def get_surface_points(self) -> np.ndarray:
        """
        Access the surface points used in the Voronoi construction.
        (If you want to keep them around, we can modify the core
        function to return them as well.)
        """
        raise NotImplementedError("Surface points not stored in this wrapper yet.")


# ---- Example usage ----
if __name__ == "__main__":
    import biotite.structure.io as strucio

    atom_array = strucio.load_structure('/Users/yusuf/6viz.cif')
    pe = PackingEntropy(atom_array, probe_size=1.4, onspherepoints=30)

    df = pe.get_dataframe()
    print(df.head())

    print("Total packing fraction:", pe.get_total_packing_fraction())
    print("Total entropy:", pe.get_total_entropy())
    print("Chain A entropy:", pe.get_total_chain_entropy("A"))
    ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]