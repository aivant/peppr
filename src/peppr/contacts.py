__all__ = [
    "ContactMeasurement",
    "find_atoms_by_pattern",
    "find_resonance_charges",
    "get_interchangeable_tautomers",
]

from enum import IntEnum
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import rdkit.Chem.AllChem as Chem
from numpy.typing import NDArray
from rdkit.Chem.MolStandardize import rdMolStandardize
from peppr.charge import estimate_formal_charges
from peppr.sanitize import sanitize

# Create a proper Python Enum for the RDKit HybridizationType
HybridizationType = IntEnum(  # type: ignore[misc]
    "HybridizationType",
    [(member.name, value) for value, member in Chem.HybridizationType.values.items()],
)

_ORIG_IDX = "_origIdx"
_ORIG_NUM_HEAVY_NEIGHS = "_origNumHeavyNeighs"

_ANGLES_FOR_HYBRIDIZATION = np.zeros(len(HybridizationType), dtype=float)
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.UNSPECIFIED] = np.nan  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.S] = np.nan  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.SP] = np.deg2rad(180)  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.SP2] = np.deg2rad(120)  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.SP3] = np.deg2rad(109.5)  # type: ignore[attr-defined]
# For d-orbitals, there are actually multiple optimal angles which are not rigorously
# checked here (see warning in docstring)
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.SP2D] = np.deg2rad(90.0)  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.SP3D2] = np.deg2rad(90.0)  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.SP3D] = np.deg2rad(90.0)  # type: ignore[attr-defined]
_ANGLES_FOR_HYBRIDIZATION[HybridizationType.OTHER] = np.nan  # type: ignore[attr-defined]


class ContactMeasurement:
    """
    This class allows measurements of receptor-ligand contacts of specific types
    (e.g. hydrogen bonds) by using *SMARTS* patterns.

    The actual measurement is performed by calling :meth:`find_contacts_by_pattern()`.

    Parameters
    ----------
    receptor, ligand : AtomArray
        The receptor and ligand to measure contacts between.
        They must only contain heavy atoms, hydrogen atoms are treated implicitly.
    cutoff : float
        The cutoff distance to use for determining the receptor binding site.
        This means a receptor atom is only taken into consideration, if it is part of
        a residue where at least one atom is within the cutoff distance to at least one
        ligand atom.
    ph : float, optional
        The pH of the environment.
        By default a physiological pH value is used [1]_.
    use_resonance : bool, optional
        If ``True``, not only explicitly charged atoms in the input receptor and
        ligand are checked, but also charged atoms that appear in their resonance
        structures.
    use_tautomers : bool = True, optional
        If ``use_tautomers`` is ``True``, the input receptor and ligand are
        expanded to include their tautomeric forms.
        This may yield a significant overhead for the binding site, as many
        tautomers may be possible for a large molecule.

    Warnings
    --------
    The specified ``cutoff`` parameter needs to be larger than the longest contact
    threshold considered for the measurement of the binding site. Additionally,
    to avoid edge effect, additional buffer of ~1.5 Å is recommended to avoid
    artificial interactions with *ionized ends*.

    Notes
    -----
    Checking resonance structures is desirable in most cases. For example, in
    a carboxyl group the charged oxygen atom might not be within the threshold
    distance of another positively charged atom, but the other oxygen atom in
    the group might be. When ``use_resonance=True``, both oxygen atoms are checked.

    Similarly, tautomer enumeration is desirable for heavy-atom-only structures,
    where the proton position is unknown. For example, a histidine side chain
    can donate or accept a hydrogen bond at either nitrogen, depending on its
    tautomeric form. When ``use_tautomers=True``, all interchangeable tautomeric
    forms are considered when matching SMARTS patterns.

    References
    ----------
    .. [1] https://www.ncbi.nlm.nih.gov/books/NBK507807/
    """

    def __init__(
        self,
        receptor: struc.AtomArray,
        ligand: struc.AtomArray,
        cutoff: float = 8.0,  # smaller not recommended for charged interactions
        ph: float = 7.4,
        use_resonance: bool = True,
        use_tautomers: bool = True,
    ):
        if np.any(receptor.element == "H") or np.any(ligand.element == "H"):
            raise ValueError("Structures must only contain heavy atoms")

        # For performance only convert the receptor binding site into a 'Mol'
        # To determine the binding site we need to find receptor atoms within cutoff
        # distance to the ligand
        receptor_cell_list = struc.CellList(receptor, cutoff)
        # We need to count each atom in the binding site only once even if multiple
        # ligand atoms are within cutoff distance to it
        binding_site_indices = np.unique(
            receptor_cell_list.get_atoms(ligand.coord, cutoff).flatten()
        )
        # Remove the padding values
        binding_site_indices = binding_site_indices[binding_site_indices != -1]
        # Get the residues containing at least one binding site atom
        residue_mask = np.any(
            struc.get_residue_masks(receptor, binding_site_indices), axis=0
        )
        binding_site = receptor[residue_mask]

        # Used for mapping back indices pointing to the binding site
        # to indices pointing to the entire receptor
        self._binding_site_indices = np.where(residue_mask)[0]
        self._binding_site = binding_site
        self._ligand = ligand.copy()

        try:
            # Detect charged atoms to find salt bridges
            # and detect molecular patterns involving charged atoms
            self._binding_site.set_annotation(
                "charge", estimate_formal_charges(self._binding_site, ph)
            )
            self._ligand.set_annotation(
                "charge", estimate_formal_charges(self._ligand, ph)
            )
        except Exception as e:
            raise struc.BadStructureError(
                "A valid molecule is required for charge estimation"
            ) from e

        # Convert to 'Mol' object to allow for matching SMARTS patterns
        self._binding_site_mol = rdkit_interface.to_mol(self._binding_site)
        self._ligand_mol = rdkit_interface.to_mol(self._ligand)
        # For matching some SMARTS strings a properly sanitized molecule is required
        sanitize(self._binding_site_mol)
        sanitize(self._ligand_mol)
        self.use_resonance = use_resonance
        self.use_tautomers = use_tautomers
        if self.use_resonance or self.use_tautomers:
            self._binding_site_residue_map = ResidueMap(self._binding_site_mol)
            self._ligand_residue_map = ResidueMap(self._ligand_mol)
        if self.use_resonance:
            (
                self._binding_site_pos_mask,
                self._binding_site_neg_mask,
                self._binding_site_conjugated_groups,
            ) = self._binding_site_residue_map.get_resonance_charges()
            (
                self._ligand_pos_mask,
                self._ligand_neg_mask,
                self._ligand_conjugated_groups,
            ) = self._ligand_residue_map.get_resonance_charges()

    def find_contacts_by_pattern(
        self,
        receptor_pattern: str,
        ligand_pattern: str,
        distance_scaling: tuple[float, float],
        receptor_ideal_angle: float | None = None,
        ligand_ideal_angle: float | None = None,
        tolerance: float = np.deg2rad(30),
    ) -> NDArray[np.int_]:
        """
        Find contacts between the receptor and ligand atoms that fulfill the given
        *SMARTS* patterns.

        Parameters
        ----------
        receptor_pattern, ligand_pattern : str
            The SMARTS pattern to match receptor and ligand atoms against, respectively.
            This means the set of atoms that can form a contact is limited to the
            matched atoms.
        distance_scaling : tuple(float, float)
            Only atoms within a certain distance range count as a contact.
            This distance range is the sum of VdW radii of the two atoms
            multiplied by the lower and upper bound scaling factor given by this
            parameter.
        receptor_ideal_angle, ligand_ideal_angle : float, optional
            If an angle (in radians) is given, this angle is used as the ideal contact
            angle.
            By default, ideal contact angle is based on hybridization state of the
            receptor and ligand atoms in contact, respectively.
        tolerance : float
            The maximum allowed deviation from an ideal contact angle, that is based
            on hybridization state.
            The angle is given in radians.

        Returns
        -------
        np.ndarray, shape=(n,2), dtype=int
            The indices of the receptor and ligand atoms that fulfill the given
            SMARTS pattern and are within the given distance range.
            The first column points to the receptor atom and the second column to the
            ligand atom.

        Warnings
        --------
        When checking for ideal contact angle (when `receptor_ideal_angle` or
        `ligand_ideal_angle` is ``None``, default), only one neighbor is used
        to calculate the orbital angle, assuming undistorted configuration.
        This is accurate for most cases, but may miss contacts if d-orbitals are
        involved in a putative contact atom.
        This is true for e.g. metal atoms.

        Notes
        -----
        The pattern must target a single atom, not a group of atoms.
        For example the pattern ``CC`` would lead to an exception, as it would match
        a group of two carbon atoms.
        """
        if self.use_tautomers:
            matched_receptor_indices = (
                self._binding_site_residue_map.find_tautomer_atoms_by_pattern(
                    receptor_pattern
                )
            )
            matched_ligand_indices = (
                self._ligand_residue_map.find_tautomer_atoms_by_pattern(ligand_pattern)
            )
        else:
            matched_receptor_indices = find_atoms_by_pattern(
                self._binding_site_mol, receptor_pattern
            )
            matched_ligand_indices = find_atoms_by_pattern(
                self._ligand_mol, ligand_pattern
            )

        combined_vdw_radii = (
            np.array(
                [
                    info.vdw_radius_single(element)
                    for element in self._binding_site.element[matched_receptor_indices]
                ]
            )[:, None]
            + np.array(
                [
                    info.vdw_radius_single(element)
                    for element in self._ligand.element[matched_ligand_indices]
                ]
            )[None, :]
        )

        # Perform the distance check
        # Create a distance matrix of all contact candidates
        distances = struc.distance(
            self._binding_site.coord[matched_receptor_indices, None],
            self._ligand.coord[None, matched_ligand_indices],
        )
        # The smaller value is always the lower bound
        lower_bound, upper_bound = sorted(distance_scaling)
        lower_bound = lower_bound * combined_vdw_radii
        upper_bound = upper_bound * combined_vdw_radii
        # Find all contacts within the given distance range
        contacts = np.where((distances >= lower_bound) & (distances <= upper_bound))
        # Note that these indices point to the already filtered down matches
        # -> map them back to indices that point to the binding site and ligand
        receptor_indices, ligand_indices = contacts
        receptor_indices = matched_receptor_indices[receptor_indices]
        ligand_indices = matched_ligand_indices[ligand_indices]

        # Perform the angle check
        ligand_angles = struc.angle(
            _get_neighbor_pos(self._ligand, ligand_indices),
            self._ligand.coord[ligand_indices],
            self._binding_site.coord[receptor_indices],
        )
        receptor_angles = struc.angle(
            _get_neighbor_pos(self._binding_site, receptor_indices),
            self._binding_site.coord[receptor_indices],
            self._ligand.coord[ligand_indices],
        )
        if ligand_ideal_angle is None:
            ligand_ideal_angle = _get_angle_to_lone_electron_pair(
                self._ligand_mol, ligand_indices
            )  # type: ignore[assignment]
        if receptor_ideal_angle is None:
            receptor_ideal_angle = _get_angle_to_lone_electron_pair(
                self._binding_site_mol, receptor_indices
            )  # type: ignore[assignment]
        is_contact = _acceptable_angle(
            ligand_angles, ligand_ideal_angle, tolerance
        ) & _acceptable_angle(receptor_angles, receptor_ideal_angle, tolerance)
        ligand_indices = ligand_indices[is_contact]
        receptor_indices = receptor_indices[is_contact]

        return np.stack(
            (
                # Furthermore, the indices pointing to the binding site need to be
                # mapped to the entire receptor indices
                self._binding_site_indices[receptor_indices],
                ligand_indices,
            ),
            axis=1,
        )

    def find_salt_bridges(
        self,
        threshold: float = 4.0,
    ) -> NDArray[np.int_]:
        """
        Find salt bridges between the receptor and ligand atoms.

        A salt bridge is a contact between two oppositely charged atoms within a certain
        threshold distance.

        Parameters
        ----------
        threshold : float, optional
            The maximum distance between two charged atoms to consider them as a salt
            bridge.
            Note that this is also constrained by the given `cutoff` in the constructor.

        Returns
        -------
        np.ndarray, shape=(n,2), dtype=int
            The indices of the receptor and ligand atoms that form a salt bridge.
            The first column points to the receptor atom and the second column to the
            ligand atom.

        Notes
        -----
        Checking resonance structures is desirable in most cases.
        For example in a carboxyl group the charged oxygen atom might not be within the
        threshold distance of another positively charged atom, but the other oxygen atom
        in the group might be.
        When ``ContactMeasurement.use_resonance=True``, both oxygen atoms are checked.
        """
        if self.use_resonance:
            binding_site_pos_indices = np.where(self._binding_site_pos_mask)[0]
            binding_site_neg_indices = np.where(self._binding_site_neg_mask)[0]
            ligand_pos_indices = np.where(self._ligand_pos_mask)[0]
            ligand_neg_indices = np.where(self._ligand_neg_mask)[0]
        else:
            ligand_pos_indices = np.where(self._ligand.charge > 0)[0]
            ligand_neg_indices = np.where(self._ligand.charge < 0)[0]
            binding_site_pos_indices = np.where(self._binding_site.charge > 0)[0]
            binding_site_neg_indices = np.where(self._binding_site.charge < 0)[0]

        bridge_indices = []
        # Try both cases where either the ligand or receptor atoms is positively charged
        for binding_site_indices, ligand_indices in [
            (binding_site_pos_indices, ligand_neg_indices),
            (binding_site_neg_indices, ligand_pos_indices),
        ]:
            # Create a distance matrix of all possible bridges...
            distances = struc.distance(
                self._binding_site.coord[binding_site_indices, None],
                self._ligand.coord[None, ligand_indices],
            )
            fulfilled_binding_site_indices, fulfilled_ligand_indices = np.where(
                distances <= threshold
            )
            # ...and check which of them fulfill the threshold criterion
            # The smaller value is always the lower bound
            bridge_indices.append(
                (
                    (binding_site_indices[fulfilled_binding_site_indices]),
                    (ligand_indices[fulfilled_ligand_indices]),
                )
            )
        # Combine the indices of both cases
        bridge_indices = np.concatenate(bridge_indices, axis=-1).T

        if self.use_resonance:
            # Remove duplicate bridges that originate from the same conjugated group
            binding_site_groups = self._binding_site_conjugated_groups[
                bridge_indices[:, 0]
            ]
            ligand_groups = self._ligand_conjugated_groups[bridge_indices[:, 1]]
            _, unique_indices = np.unique(
                np.stack((binding_site_groups, ligand_groups), axis=1),
                axis=0,
                return_index=True,
            )
            bridge_indices = bridge_indices[unique_indices]

        # Map indices pointing to the binding site
        # to indices pointing to the entire receptor
        bridge_indices[:, 0] = self._binding_site_indices[bridge_indices[:, 0]]
        return bridge_indices

    def find_stacking_interactions(
        self,
        threshold: float = 6.5,
        plane_angle_tol: float = np.deg2rad(30),
        shift_angle_tol: float = np.deg2rad(30),
    ) -> list[tuple[NDArray[np.int_], NDArray[np.int_], struc.PiStacking]]:
        """
        Find π-stacking interactions between aromatic rings across the binding interface.

        Wrapper around biotite.structure.find_stacking_interactions that filters for
        interactions between binding site and ligand only.

        Parameters
        ----------
        threshold : float, optional
            The cutoff distance for ring centroids.
        plane_angle_tol : float, optional
            Tolerance for angle between ring planes (radians).
        shift_angle_tol : float, optional
            Tolerance for angle between ring normals and centroid vector (radians).

        Returns
        -------
        list of tuple
            Each tuple contains
            ``(binding_site_ring_indices, ligand_ring_indices, stacking_type)``.
        """
        combined_atoms = self._binding_site + self._ligand
        all_interactions = struc.find_stacking_interactions(
            combined_atoms, threshold, plane_angle_tol, shift_angle_tol
        )

        return [
            self._map_stacking_indices(indices_1, indices_2, kind)
            for indices_1, indices_2, kind in all_interactions
            # filter out intra polymer and intra ligand interactions
            if (indices_1[0] >= len(self._binding_site))
            != (indices_2[0] >= len(self._binding_site))
        ]

    def _map_stacking_indices(
        self,
        indices_1: NDArray[np.int_],
        indices_2: NDArray[np.int_],
        kind: struc.PiStacking,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], struc.PiStacking]:
        """Map combined structure indices back to original receptor and ligand."""
        # Sort to ensure polymer indices comes first then ligand
        polymer_idx, ligand_idx = sorted(
            [indices_1, indices_2], key=lambda idx: idx[0] >= len(self._binding_site)
        )

        return (
            self._binding_site_indices[polymer_idx],  # Map to full receptor
            ligand_idx - len(self._binding_site),  # Map to ligand
            kind,
        )

    def find_pi_cation_interactions(
        self,
        distance_cutoff: float = 5.0,
        angle_tol: float = np.deg2rad(30.0),
    ) -> list[tuple[NDArray[np.int_], NDArray[np.int_], bool]]:
        """
        Find π-cation interactions between aromatic rings and cations across the binding
        interface.

        Wrapper around :func:`biotite.structure.find_pi_cation_interactions()` that
        filters for interactions between binding site and ligand only.

        Parameters
        ----------
        distance_cutoff : float, optional
            The cutoff distance between ring centroid and cation.
        angle_tol : float, optional
            The tolerance for the angle between the ring plane normal
            and the centroid-cation vector. Perfect pi-cation interaction
            has 0° angle (perpendicular to ring plane).
            Given in radians.

        Returns
        -------
        list of tuple
            Each tuple contains
            ``(receptor_indices, ligand_indices, cation_in_receptor)``.
            ``cation_in_receptor`` is ``True``, if the interacting cation is in the
            receptor molecules and ``False`` otherwise.
        """
        combined_atoms = self._binding_site + self._ligand
        if self.use_resonance:
            # consider all positively charged atoms in resonance
            pos_mask = np.concatenate(
                [self._binding_site_pos_mask, self._ligand_pos_mask], axis=0
            )
            # for this function is enough to consider just positive atoms
            combined_atoms.charge = pos_mask.astype(int)
        binding_site_size = len(self._binding_site)
        all_interactions = struc.find_pi_cation_interactions(
            combined_atoms, distance_cutoff, angle_tol
        )
        cross_interactions = [
            (ring_indices, cation_index)
            for ring_indices, cation_index in all_interactions
            # filter out intra polymer and intra ligand interactions
            if (ring_indices[0] >= binding_site_size)
            != (cation_index >= binding_site_size)
        ]

        if self.use_resonance and cross_interactions:
            # Remove duplicate interactions where cations from the same
            # conjugated group interact with the same ring
            conjugated_groups = np.concatenate(
                [
                    self._binding_site_conjugated_groups,
                    self._ligand_conjugated_groups,
                ]
            )
            # use the first ring atom index in place of unique ring id
            ring_starts = np.array([ri[0] for ri, _ in cross_interactions])
            cation_groups = conjugated_groups[[ci for _, ci in cross_interactions]]
            _, unique_indices = np.unique(
                np.stack((ring_starts, cation_groups), axis=1),
                axis=0,
                return_index=True,
            )
            cross_interactions = [cross_interactions[i] for i in unique_indices]

        return [
            self._map_pi_cation_indices(ring_indices, cation_index)
            for ring_indices, cation_index in cross_interactions
        ]

    def _map_pi_cation_indices(
        self, ring_indices: NDArray[np.int_], cation_index: int
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], bool]:
        """
        Map combined structure indices for a pi-cation interaction back to the
        original receptor and ligand.
        """
        binding_site_size = self._binding_site.array_length()
        cation_in_receptor = cation_index < binding_site_size

        if cation_in_receptor:
            # Case 2: Cation in receptor, Ring in ligand
            receptor_indices = np.array(
                [self._binding_site_indices[cation_index]], dtype=int
            )
            ligand_indices = ring_indices - binding_site_size
        else:
            # Case 1: Ring in receptor, Cation in ligand
            receptor_indices = self._binding_site_indices[ring_indices]
            ligand_indices = np.array([cation_index - binding_site_size], dtype=int)

        return (receptor_indices, ligand_indices, cation_in_receptor)


def find_atoms_by_pattern(
    mol: Chem.Mol,
    pattern: str,
) -> NDArray[np.int_]:
    """
    Find atoms that fulfill the given SMARTS pattern.

    Parameters
    ----------
    mol : Mol
        The atoms to find matches for.
    pattern : str
        The SMARTS pattern to match against.

    Returns
    -------
    np.ndarray, shape=(n,), dtype=int
        The atom indices that fulfill the given SMARTS pattern.
    """
    pattern = Chem.MolFromSmarts(pattern)  # type: ignore[attr-defined]
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        if len(match) > 1:
            raise ValueError(
                "The given pattern must target only one atom, not a group of atoms"
            )
    # Remove that last dimension as otherwise the shape would be (n, 1),
    # as only a single matched atom per match is allowed (as asserted above)
    return np.array(matches, dtype=int).flatten()


def _get_neighbor_pos(
    atoms: struc.AtomArray, indices: NDArray[np.int_]
) -> NDArray[np.floating]:
    """
    Get the coordinates of the respective neighbors of the given atoms.
    If an atom has multiple neighbors, one of them is arbitrarily chosen.

    Parameters
    ----------
    atoms : AtomArray
        The structure containing all atoms.
    indices : ndarray, shape=(n,), dtype=int
        The indices of the atoms to get the neighbor positions for.

    Returns
    -------
    vectors : ndarray, shape=(n,3), dtype=float
        The coordinates of the respective neighbors of the given atoms.
    """
    all_bonds, _ = atoms.bonds.get_all_bonds()
    if all_bonds.shape[1] == 0:
        # No atom has any neighbor (i.e. an empty BondList)
        # -> getting the first neighbor below would lead to an IndexError
        return np.full((len(indices), 3), np.nan)
    neighbor_indices = all_bonds[indices]
    # Arbitrarily choose the first neighbor
    neighbor_coord = atoms.coord[neighbor_indices[:, 0]]
    # Handle the case where an atom has no neighbor
    neighbor_coord[neighbor_indices[:, 0] == -1] = np.nan
    return neighbor_coord


def _get_angle_to_lone_electron_pair(
    mol: Chem.Mol,
    indices: NDArray[np.int_],
) -> NDArray[np.floating]:
    """
    Get the angle between the neighbor and the lone electron pair of the indexed atoms.

    Note: the `lone electron pair` is misnomer for hb donors as it would position the
    implicit hydrogen location.

    Parameters
    ----------
    mol : Mol
        The molecule containing the atoms.
    indices : ndarray, shape=(n,), dtype=int
        The indices of the atoms to get the angle for.

    Returns
    -------
    angles : ndarray, shape=(n,), dtype=float
        The angle between the neighbor and the lone electron pair of the indexed atoms.
    """
    return _ANGLES_FOR_HYBRIDIZATION[
        [mol.GetAtomWithIdx(i.item()).GetHybridization() for i in indices]
    ]


def _acceptable_angle(
    angle: NDArray[np.floating] | float,
    ref_angle: NDArray[np.floating] | float,
    tolerance: float,
) -> NDArray[np.bool_] | bool:
    """
    Check if a given angle is within a certain tolerance of the ideal angle.
    The angle is given in radians.

    Parameters
    ----------
    angle : float
        The angle to check.
    ref_angle : float
        The ideal angle.
    tolerance : float
        The tolerance to use.

    Returns
    -------
    is_acceptable : bool
        If the angle is within the tolerance of the ideal angle.
    """
    return abs(angle - ref_angle) <= tolerance  # type: ignore[operator]


class ResidueMap:
    """
    Per-residue split of a molecule with methods for resonance and
    tautomer analysis.

    For performance reasons the input `Mol` is split into individual
    residues.
    Therefore no cross-residue resonance or tautomeric interaction
    can be detected, i.e. conjugation and proton transfer do not cross
    peptide bonds.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to split by PDB/mmCIF residue names.

    Notes
    -----
    Splitting a molecule at inter-residue bonds creates artificial
    fragment termini. The lost heavy-atom neighbors get perceived as
    phantom implicit hydrogens to satisfy valence (e.g. PRO backbone N
    goes from three heavy-atom bonds to two, picking up an implicit H
    that was never present in the intact molecule).

    To detect these artifacts, the original heavy-atom neighbor count
    (``_origNumHeavyNeighs = totalDegree - totalHs``) is recorded on
    every atom *before* splitting. In
    :meth:`find_tautomer_atoms_by_pattern`, any match on an atom whose
    current heavy-neighbor count is below the recorded value *and* that
    carries exactly one hydrogen is discarded as a false positive.
    """

    def __init__(self, mol: Chem.Mol) -> None:
        self._n_atoms = mol.GetNumAtoms()
        mol = Chem.RWMol(mol)  # avoid mutating the caller's molecule

        # Record original index and heavy-atom neighbor count before
        # splitting, so downstream code can detect phantom Hs.
        for i in range(self._n_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom.SetIntProp(_ORIG_IDX, i)
            atom.SetIntProp(
                _ORIG_NUM_HEAVY_NEIGHS,
                atom.GetTotalDegree() - atom.GetTotalNumHs(),
            )

        # Two-level split: by res name, then into individual instances
        residues = Chem.rdmolops.SplitMolByPDBResidues(mol)

        self._instances: list[Chem.Mol] = []
        self._index_maps: list[NDArray[np.int_]] = []
        for group in residues.values():
            for instance in Chem.rdmolops.GetMolFrags(group, asMols=True):
                idx_map = np.array(
                    [
                        instance.GetAtomWithIdx(i).GetIntProp(_ORIG_IDX)
                        for i in range(instance.GetNumAtoms())
                    ],
                    dtype=int,
                )
                self._index_maps.append(idx_map)
                sanitize(instance)
                self._instances.append(instance)

        # define a placeholder for tautomers
        self._tautomers: list[list[Chem.Mol]] | None = None

    def get_resonance_charges(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int_]]:
        """
        Compute resonance charges per residue and assemble into
        molecule-wide arrays.

        Returns
        -------
        pos_mask : ndarray, shape=(n,), dtype=bool
            Mask of positively charged atoms across all resonance
            forms.
        neg_mask : ndarray, shape=(n,), dtype=bool
            Mask of negatively charged atoms across all resonance
            forms.
        conjugated_groups : ndarray, shape=(n,), dtype=int
            Conjugated group id for each atom. Atoms in the same
            group share a single delocalised charge.
        """
        pos_mask = np.zeros(self._n_atoms, dtype=bool)
        neg_mask = np.zeros(self._n_atoms, dtype=bool)
        conjugated_groups = np.full(self._n_atoms, -1, dtype=int)
        group_offset = 0

        for instance, idx_map in zip(self._instances, self._index_maps):
            pos, neg, groups = find_resonance_charges(instance)
            pos_mask[idx_map] |= pos
            neg_mask[idx_map] |= neg
            conjugated_groups[idx_map] = groups + group_offset
            group_offset = conjugated_groups.max() + 1

        return pos_mask, neg_mask, conjugated_groups

    def _ensure_tautomers(self) -> None:
        """Lazily enumerate tautomers (once per unique SMILES)."""
        if self._tautomers is not None:
            # do not repeat if done already
            return

        # cache is local to this function
        cache: dict[str, list[Chem.Mol]] = {}
        self._tautomers = []
        for instance in self._instances:
            key = Chem.MolToSmiles(instance)  # type: ignore[attr-defined]
            if key not in cache:
                cache[key] = get_interchangeable_tautomers(instance)
            self._tautomers.append(cache[key])

    def find_tautomer_atoms_by_pattern(self, pattern: str) -> NDArray[np.int_]:
        """
        Find atoms matching a SMARTS pattern across all tautomeric
        forms of each residue instance, mapped back to the original
        molecule's atom indices.

        Atoms that lost a heavy-atom neighbor during splitting and
        carry exactly one hydrogen are excluded, as that hydrogen is
        a phantom artifact of the lost bond (e.g. PRO backbone N
        gaining an implicit H after losing the peptide bond).

        Parameters
        ----------
        pattern : str
            The SMARTS pattern to match against.

        Returns
        -------
        np.ndarray, shape=(n,), dtype=int
            Sorted atom indices (in the original molecule) that match
            the pattern in any tautomeric form.
        """
        self._ensure_tautomers()
        matches: set[np.int_] = set()
        for tautomers, idx_map in zip(self._tautomers, self._index_maps):
            for tautomer in tautomers:
                local_hits = find_atoms_by_pattern(tautomer, pattern)
                for hit in local_hits:
                    atom = tautomer.GetAtomWithIdx(int(hit))
                    # Skip false-positive donors: atoms that lost a
                    # heavy neighbor during splitting and have exactly
                    # 1 H (the phantom H replacing the lost bond).
                    if (
                        atom.HasProp(_ORIG_NUM_HEAVY_NEIGHS)
                        and atom.GetTotalNumHs() == 1
                        and (atom.GetTotalDegree() - atom.GetTotalNumHs())
                        < atom.GetIntProp(_ORIG_NUM_HEAVY_NEIGHS)
                    ):
                        continue
                    matches.add(idx_map[hit])
        return np.array(np.sort(list(matches)), dtype=int).flatten()


def find_resonance_charges(
    mol: Chem.Mol,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int_]]:
    """
    Find indices of positively and negatively charged atoms in the given molecule
    and its resonance structures.

    Parameters
    ----------
    mol : Mol
        The molecule to find the charged atoms in.

    Returns
    -------
    pos_mask : ndarray, shape=(n,), dtype=bool
        The mask of positively charged atoms.
    neg_mask : ndarray, shape=(n,), dtype=bool
        The mask of negatively charged atoms.
    conjugated_groups : ndarray, shape=(n,), dtype=int
        The *conjugated group* for each atoms.
        Atoms from the same group are denoted by the same integer.
        This means that a single charge may appear multiple times in `pos_mask` or
        `neg_mask`, as the corresponding atoms are part of the same conjugated group.

    Warnings
    --------
    RDKit's ResonanceMolSupplier only shuffles existing formal charges through
    conjugated systems. It does not generate zwitterionic/charge-separated
    resonance forms from neutral species.
    """
    pos_mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    neg_mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    if not any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        # if no charges present - no need to look for resonance
        conjugated_groups = np.full(mol.GetNumAtoms(), -1, dtype=int)
        return pos_mask, neg_mask, conjugated_groups

    resonance_supplier = Chem.ResonanceMolSupplier(mol)
    for resonance_mol in resonance_supplier:
        if resonance_mol is None:
            raise struc.BadStructureError("Cannot compute resonance structure")
        for i in range(mol.GetNumAtoms()):
            charge = resonance_mol.GetAtomWithIdx(i).GetFormalCharge()
            if charge > 0:
                pos_mask[i] = True
            elif charge < 0:
                neg_mask[i] = True
    conjugated_groups = np.array(
        [resonance_supplier.GetAtomConjGrpIdx(i) for i in range(mol.GetNumAtoms())],
        dtype=int,
    )
    # Assign each non-conjugated atom to a unique group
    non_conjugated_mask = conjugated_groups == -1
    # Handle edge case that the given molecule has no atoms
    max_group = np.max(conjugated_groups) if len(conjugated_groups) > 0 else 0
    conjugated_groups[non_conjugated_mask] = np.arange(
        max_group + 1,
        max_group + 1 + np.count_nonzero(non_conjugated_mask),
        dtype=int,
    )
    return pos_mask, neg_mask, conjugated_groups


def get_interchangeable_tautomers(mol: Chem.Mol) -> list[Chem.Mol]:
    """
    Find all tautomeric forms of the molecule that are on par (or better) forms
    of the input form, explicitly checking for atom hybridizations.

    Molecules can exist in multiple tautomeric forms, which can differ in the
    position of protons and double bonds, yet retaining same atom hybridization.
    As a result, with heavy atom only description, it is impossible to tell
    these forms apart.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to find the tautomeric forms for each fragment.

    Returns
    -------
    list[Chem.Mol]
        Array of explicit primary tautomeric forms of the molecule.

    Warnings
    --------
    This uses RDKit defaults for MaxTautomers=1000 and MaxTransforms=1000, that
    limits the search exhaustiveness to reduce the runtime. Consider applying
    this function on molecule fragments to improve efficiency.
    """
    tautomerator = rdMolStandardize.TautomerEnumerator()
    # make an editable copy of the original molecule to avoid modifying input
    mol = Chem.RWMol(mol)
    # keep original hybridization - this should not change!
    orig_hybridization = np.array([at.GetHybridization() for at in mol.GetAtoms()])
    # add labels to break symmetry to force consider symmetric tautomers as unique
    for i, at in enumerate(mol.GetAtoms()):
        at.SetAtomMapNum(i + 1)
    tauts = np.array(tautomerator.Enumerate(mol))
    # find the best tautomers according to the RDKit scoring function and only keep those
    # that are on par or better than the input form
    tautomer_scores = np.array([tautomerator.ScoreTautomer(taut) for taut in tauts])
    good_tauts = np.where(tautomer_scores >= tautomerator.ScoreTautomer(mol))[0]
    # get final tautomers
    final_tautomers = []
    for tmol in tauts[good_tauts]:
        # keep only the ones that keep same atom hybridizations!
        if (
            np.array([at.GetHybridization() for at in tmol.GetAtoms()])
            == orig_hybridization
        ).all():
            # remove labels for the returned tautomers
            for at in tmol.GetAtoms():
                at.SetAtomMapNum(0)
            final_tautomers.append(tmol)
    return final_tautomers
