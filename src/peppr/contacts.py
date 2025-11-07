__all__ = [
    "ContactMeasurement",
    "find_atoms_by_pattern",
]

from enum import IntEnum
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import rdkit.Chem.AllChem as Chem
from numpy.typing import NDArray
from peppr.charge import estimate_formal_charges
from peppr.sanitize import sanitize

# Create a proper Python Enum for the RDKit HybridizationType
HybridizationType = IntEnum(  # type: ignore[misc]
    "HybridizationType",
    [(member.name, value) for value, member in Chem.HybridizationType.values.items()],
)

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

    The actual measurement is performed by calling :func:`find_contacts_by_pattern()`.

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

    References
    ----------
    .. [1] https://www.ncbi.nlm.nih.gov/books/NBK507807/
    """

    def __init__(
        self,
        receptor: struc.AtomArray,
        ligand: struc.AtomArray,
        cutoff: float,
        ph: float = 7.4,
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
        except Exception:
            raise struc.BadStructureError(
                "A valid molecule is required for charge estimation"
            )

        # Convert to 'Mol' object to allow for matching SMARTS patterns
        self._binding_site_mol = rdkit_interface.to_mol(self._binding_site)
        self._ligand_mol = rdkit_interface.to_mol(self._ligand)
        # For matching some SMARTS strings a properly sanitized molecule is required
        sanitize(self._binding_site_mol)
        sanitize(self._ligand_mol)

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
        matched_receptor_indices = find_atoms_by_pattern(
            self._binding_site_mol, receptor_pattern
        )
        matched_ligand_indices = find_atoms_by_pattern(self._ligand_mol, ligand_pattern)

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
        use_resonance: bool = True,
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
        use_resonance : bool, optional
            If ``True``, not only explicitly charged atoms in the input receptor and
            ligand are checked, but also charged atoms that appear in their resonance
            structures.
            However, if multiple atoms from the same conjugated group form the same
            salt bridge, only one of them is kept.

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
        When ``use_resonance=True``, Both oxygen atoms would be checked.
        """
        if use_resonance:
            pos_mask, neg_mask, binding_site_conjugated_groups = (
                _find_charged_atoms_in_resonance_structures(self._binding_site_mol)
            )
            binding_site_pos_indices = np.where(pos_mask)[0]
            binding_site_neg_indices = np.where(neg_mask)[0]
            pos_mask, neg_mask, ligand_conjugated_groups = (
                _find_charged_atoms_in_resonance_structures(self._ligand_mol)
            )
            ligand_pos_indices = np.where(pos_mask)[0]
            ligand_neg_indices = np.where(neg_mask)[0]
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

        if use_resonance:
            # Remove duplicate bridges that originate from the same conjugated group
            binding_site_groups = binding_site_conjugated_groups[bridge_indices[:, 0]]
            ligand_groups = ligand_conjugated_groups[bridge_indices[:, 1]]
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
            self._map_stacking_indices(combined_atoms, indices_1, indices_2, kind)
            for indices_1, indices_2, kind in all_interactions
            # filter out intra polymer and intra ligand interactions
            if combined_atoms.hetero[indices_1[0]]
            != combined_atoms.hetero[indices_2[0]]
        ]

    def _map_stacking_indices(
        self,
        combined_atoms: struc.AtomArray,
        indices_1: NDArray[np.int_],
        indices_2: NDArray[np.int_],
        kind: struc.PiStacking,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], struc.PiStacking]:
        """Map combined structure indices back to original receptor and ligand."""
        # Sort to ensure polymer indices comes first then ligand
        polymer_idx, ligand_idx = sorted(
            [indices_1, indices_2], key=lambda idx: combined_atoms.hetero[idx[0]]
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
        binding_site_size = len(self._binding_site)
        all_interactions = struc.find_pi_cation_interactions(
            combined_atoms, distance_cutoff, angle_tol
        )

        return [
            self._map_pi_cation_indices(ring_indices, cation_index)
            for ring_indices, cation_index in all_interactions
            #  filter out intra polymer and intra ligand interactions
            if (ring_indices[0] >= binding_site_size)
            != (cation_index >= binding_site_size)
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
    angle: float,
    ref_angle: float,
    tolerance: float,
) -> bool:
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
    return abs(angle - ref_angle) <= tolerance


def _find_charged_atoms_in_resonance_structures(
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
    """
    pos_mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    neg_mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
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
    try:
        conjugated_groups = np.array(
            [resonance_supplier.GetAtomConjGrpIdx(i) for i in range(mol.GetNumAtoms())],
            dtype=int,
        )
    except RuntimeError:
        # This is a bug in RDKit, that happens if the molecule has no bonds at all
        # (https://github.com/rdkit/rdkit/issues/8638)
        conjugated_groups = np.full(mol.GetNumAtoms(), -1, dtype=int)
    # Fix the 'integer underflow issue' for -1 values
    # (https://github.com/rdkit/rdkit/issues/7112)
    conjugated_groups[conjugated_groups == 2**32 - 1] = -1
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
