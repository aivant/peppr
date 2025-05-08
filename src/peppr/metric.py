__all__ = [
    "Metric",
    "MonomerRMSD",
    "MonomerTMScore",
    "MonomerLDDTScore",
    "IntraLigandLDDTScore",
    "LDDTPLIScore",
    "LDDTPPIScore",
    "GlobalLDDTScore",
    "DockQScore",
    "LigandRMSD",
    "InterfaceRMSD",
    "ContactFraction",
    "PocketAlignedLigandRMSD",
    "BiSyRMSD",
    "BondLengthViolations",
    "BondAngleViolations",
    "ClashCount",
]

import itertools
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple
import biotite.structure as struc
import numpy as np
from numpy.typing import NDArray
from peppr.bisyrmsd import bisy_rmsd
from peppr.clashes import find_clashes
from peppr.common import is_small_molecule
from peppr.dockq import (
    dockq,
    fnat,
    get_contact_residues,
    irmsd,
    lrmsd,
    pocket_aligned_lrmsd,
)
from peppr.graph import graph_to_connected_triples
from peppr.idealize import idealize_bonds


class Metric(ABC):
    """
    The base class for all evaluation metrics.

    The central :meth:`evaluate()` method takes a for a system reference and pose
    structures as input and returns a sclar score.

    Attributes
    ----------
    name : str
        The name of the metric.
        Used for displaying the results via the :class:`Evaluator`.
        **ABSTRACT:** Must be overridden by subclasses.
    thresholds : dict (str -> float)
        The named thresholds for the metric.
        Each threshold contains the lower bound
    """

    def __init__(self) -> None:
        thresholds = list(self.thresholds.values())
        if sorted(thresholds) != thresholds:
            raise ValueError("Thresholds must be sorted in ascending order")

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict()

    @abstractmethod
    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Apply this metric on the given predicted pose with respect to the given
        reference.

        **ABSTRACT:** Must be overridden by subclasses.

        Parameters
        ----------
        reference : AtomArray, shape=(n,)
            The reference structure of the system.
            Each separate instance/molecule must have a distinct `chain_id`.
        pose : AtomArray, shape=(n,)
            The predicted pose.
            Must have the same length and atom order as the `reference`.

        Returns
        -------
        float
            The metric computed for each pose.
            *NaN*, if the structure is not suitable for this metric.

        Notes
        -----
        Missing atoms in either the reference or the pose can be identified with
        *NaN* values.
        """
        raise NotImplementedError

    @abstractmethod
    def smaller_is_better(self) -> bool:
        """
        Whether as smaller value of this metric is considered a better prediction.

        **ABSTRACT:** Must be overridden by subclasses.

        Returns
        -------
        bool
            If true, a smaller value of this metric is considered a better prediction.
            Otherwise, a larger value is considered a better prediction.
        """
        raise NotImplementedError


class MonomerRMSD(Metric):
    r"""
    Compute the *root mean squared deviation* (RMSD) between each peptide chain in the
    reference and the pose and take the mean weighted by the number of heavy atoms.

    Parameters
    ----------
    threshold : float
        The RMSD threshold to use for the *good* predictions.
    ca_only : bool, optional
        If ``True``, only consider :math:`C_{\alpha}` atoms.
        Otherwise, consider all heavy atoms.
    """

    def __init__(self, threshold: float, ca_only: bool = True) -> None:
        self._threshold = threshold
        self._ca_only = ca_only
        super().__init__()

    @property
    def name(self) -> str:
        if self._ca_only:
            return "CA-RMSD"
        else:
            return "all-atom RMSD"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [(f"<{self._threshold}", 0), (f">{self._threshold}", self._threshold)]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def superimpose_and_rmsd(reference_chain, pose_chain):  # type: ignore[no-untyped-def]
            pose_chain, _ = struc.superimpose(reference_chain, pose_chain)
            return struc.rmsd(reference_chain, pose_chain)

        mask = ~reference.hetero
        if self._ca_only:
            mask &= reference.atom_name == "CA"
        return _run_for_each_monomer(reference[mask], pose[mask], superimpose_and_rmsd)

    def smaller_is_better(self) -> bool:
        return True


class MonomerTMScore(Metric):
    """
    Compute the *TM-score* score for each monomer and take the mean weighted by the
    number of atoms.
    """

    @property
    def name(self) -> str:
        return "TM-score"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def superimpose_and_tm_score(reference, pose):  # type: ignore[no-untyped-def]
            # Use 'superimpose_structural_homologs()' instead of 'superimpose()',
            # as it optimizes the TM-score instead of the RMSD
            try:
                super, _, ref_i, pose_i = struc.superimpose_structural_homologs(
                    reference, pose, max_iterations=1
                )
                return struc.tm_score(reference, super, ref_i, pose_i)
            except ValueError as e:
                if "No anchors found" in str(e):
                    # The structures are too dissimilar for structure-based
                    # superimposition, i.e. the pose is very bad
                    return 0.0
                else:
                    raise

        # TM-score is only defined for peptide chains
        mask = struc.filter_amino_acids(reference) & ~reference.hetero
        reference = reference[mask]
        pose = pose[mask]
        return _run_for_each_monomer(reference, pose, superimpose_and_tm_score)

    def smaller_is_better(self) -> bool:
        return False


class MonomerLDDTScore(Metric):
    """
    Compute the *local Distance Difference Test* (lDDT) score for each monomer
    and take the mean weighted by the number of atoms.
    """

    @property
    def name(self) -> str:
        return "intra protein lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        protein_mask = ~reference.hetero
        if not protein_mask.any():
            # No protein present
            return np.nan
        reference = reference[protein_mask]
        pose = pose[protein_mask]
        return _run_for_each_monomer(reference, pose, struc.lddt)

    def smaller_is_better(self) -> bool:
        return False


class IntraLigandLDDTScore(Metric):
    """
    Compute the *local Distance Difference Test* (lDDT) score for contacts within each
    small molecule.
    """

    @property
    def name(self) -> str:
        return "intra ligand lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def within_same_molecule(contacts: np.ndarray) -> np.ndarray:
            # Find the index of the chain/molecule for each atom
            chain_indices = struc.get_chain_positions(
                reference, contacts.flatten()
            ).reshape(contacts.shape)
            # Remove contacts between atoms of different molecules
            return chain_indices[:, 0] == chain_indices[:, 1]

        ligand_mask = reference.hetero
        if not ligand_mask.any():
            # No ligands present
            return np.nan
        reference = reference[ligand_mask]
        pose = pose[ligand_mask]
        return struc.lddt(
            reference,
            pose,
            exclude_same_residue=False,
            filter_function=within_same_molecule,
        ).item()

    def smaller_is_better(self) -> bool:
        return False


class LDDTPLIScore(Metric):
    """
    Compute the CASP LDDT-PLI score, i.e. the lDDT for protein-ligand interactions
    as defined by [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.26601
    """

    @property
    def name(self) -> str:
        return "protein-ligand lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def lddt_pli(reference, pose):  # type: ignore[no-untyped-def]
            ligand_mask = reference.hetero
            polymer_mask = ~ligand_mask

            if not polymer_mask.any():
                # No protein present -> metric is undefined
                return np.nan

            binding_site_contacts = np.unique(
                get_contact_residues(
                    reference[polymer_mask], reference[ligand_mask], cutoff=4.0
                )[:, 0]
            )
            # No contacts between the ligand and protein in reference -> no binding site
            # -> metric is undefined
            if len(binding_site_contacts) == 0:
                return np.nan
            binding_site_mask = struc.get_residue_masks(
                reference, binding_site_contacts
            ).any(axis=0)
            return struc.lddt(
                reference,
                pose,
                atom_mask=ligand_mask,
                partner_mask=binding_site_mask,
                inclusion_radius=6.0,
                distance_bins=(0.5, 1.0, 2.0, 4.0),
                symmetric=True,
            )

        return _average_over_ligands(reference, pose, lddt_pli)

    def smaller_is_better(self) -> bool:
        return False


class LDDTPPIScore(Metric):
    """
    Compute the the lDDT for protein-protein interactions, i.e. all intra-chain
    contacts are not included.
    """

    @property
    def name(self) -> str:
        return "protein-protein lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        protein_mask = ~reference.hetero
        if not protein_mask.any():
            # This is not a PPI system
            return np.nan
        reference = reference[protein_mask]
        pose = pose[protein_mask]
        return struc.lddt(reference, pose, exclude_same_chain=True)

    def smaller_is_better(self) -> bool:
        return False


class GlobalLDDTScore(Metric):
    r"""
    Compute the lDDT score for all contacts in the system, i.e. both intra- and
    inter-chain contacts.
    This is equivalent to the original lDDT definition in [1]_.

    Parameters
    ----------
    backbone_only : bool, optional
        If ``True``, only consider :math:`C_{\alpha}` from peptides and :math:`C_3^'`
        from nucleic acids.
        Otherwise, consider all heavy atoms.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btt473
    """

    def __init__(self, backbone_only: bool = True) -> None:
        self._backbone_only = backbone_only
        super().__init__()

    @property
    def name(self) -> str:
        if self._backbone_only:
            return "global backbone lDDT"
        else:
            return "global all-atom lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        if self._backbone_only:
            mask = ~reference.hetero & np.isin(reference.atom_name, ["CA", "C3'"])
            reference = reference[mask]
            pose = pose[mask]
        if reference.array_length() == 0:
            return np.nan
        return struc.lddt(reference, pose).item()

    def smaller_is_better(self) -> bool:
        return False


class DockQScore(Metric):
    """
    Compute the *DockQ* score for the given complex as defined in [1]_.

    Parameters
    ----------
    include_pli : bool, optional
        If set to ``False``, small molecules are excluded from the calculation.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    def __init__(self, include_pli: bool = True) -> None:
        self._include_pli = include_pli
        super().__init__()

    @property
    def name(self) -> str:
        if self._include_pli:
            return "DockQ"
        else:
            return "DockQ-PPI"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [
                ("incorrect", 0.0),
                ("acceptable", 0.23),
                ("medium", 0.49),
                ("high", 0.80),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def run_dockq(reference_chain1, reference_chain2, pose_chain1, pose_chain2):  # type: ignore[no-untyped-def]
            if not self._include_pli and (
                is_small_molecule(reference_chain1)
                or is_small_molecule(reference_chain2)
            ):
                # Do not compute DockQ for PLI pairs if disabled
                return np.nan
            if is_small_molecule(reference_chain1) and is_small_molecule(
                reference_chain2
            ):
                # Do not compute DockQ for small molecule pairs
                return np.nan
            return dockq(
                *_select_receptor_and_ligand(
                    reference_chain1, reference_chain2, pose_chain1, pose_chain2
                )
            ).score

        return _run_for_each_chain_pair(reference, pose, run_dockq)

    def smaller_is_better(self) -> bool:
        return False


class LigandRMSD(Metric):
    """
    Compute the *Ligand RMSD* for the given protein complex as defined in [1]_.
    The score is first separately computed for all pairs of chains that are in contact,
    and the averaged. If the reference doesn't contain any chains in contact, *NaN* is returned.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "LRMSD"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        mask = struc.filter_amino_acids(reference) & ~reference.hetero
        reference = reference[mask]
        pose = pose[mask]

        def lrmsd_on_interfaces_only(
            reference_chain1: struc.AtomArray,
            reference_chain2: struc.AtomArray,
            pose_chain1: struc.AtomArray,
            pose_chain2: struc.AtomArray,
        ) -> float | np.floating | NDArray[np.floating]:
            reference_contacts = get_contact_residues(
                reference_chain1,
                reference_chain2,
                cutoff=10.0,
            )

            if len(reference_contacts) == 0:
                return np.nan
            else:
                return lrmsd(
                    *_select_receptor_and_ligand(
                        reference_chain1, reference_chain2, pose_chain1, pose_chain2
                    )
                )

        return _run_for_each_chain_pair(reference, pose, lrmsd_on_interfaces_only)

    def smaller_is_better(self) -> bool:
        return True


class InterfaceRMSD(Metric):
    """
    Compute the *Interface RMSD* for the given protein complex as defined in [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "iRMSD"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        mask = struc.filter_amino_acids(reference) & ~reference.hetero
        reference = reference[mask]
        pose = pose[mask]
        # iRMSD is independent of the selection of receptor and ligand chain
        return _run_for_each_chain_pair(reference, pose, irmsd)

    def smaller_is_better(self) -> bool:
        return True


class ContactFraction(Metric):
    """
    Compute the fraction of correctly predicted reference contacts (*Fnat*) as defined
    in [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "fnat"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        mask = struc.filter_amino_acids(reference) & ~reference.hetero
        reference = reference[mask]
        pose = pose[mask]
        # Fnat is independent of the selection of receptor and ligand chain
        # Caution: `fnat()` returns both fnat and fnonnat -> select first element
        return _run_for_each_chain_pair(reference, pose, lambda *args: fnat(*args)[0])

    def smaller_is_better(self) -> bool:
        return False


class PocketAlignedLigandRMSD(Metric):
    """
    Compute the *Pocket aligned ligand RMSD* for the given PLI complex as defined
    in [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "PLI-LRMSD"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def run_lrmd(reference_chain1, reference_chain2, pose_chain1, pose_chain2):  # type: ignore[no-untyped-def]
            n_small_molecules = sum(
                is_small_molecule(chain)
                for chain in [reference_chain1, reference_chain2]
            )
            if n_small_molecules != 1:
                # Either two proteins or two small molecules -> not a valid PLI pair
                return np.nan
            return pocket_aligned_lrmsd(
                *_select_receptor_and_ligand(
                    reference_chain1, reference_chain2, pose_chain1, pose_chain2
                )
            )

        return _run_for_each_chain_pair(reference, pose, run_lrmd)

    def smaller_is_better(self) -> bool:
        return True


class BiSyRMSD(Metric):
    """
    Compute the *Binding-Site Superposed, Symmetry-Corrected Pose RMSD* (BiSyRMSD) for
    the given PLI complex.

    The method and default parameters are described in [1]_.

    Parameters
    ----------
    threshold : float
        The RMSD threshold to use for the *good* predictions.
    inclusion_radius : float, optional
        All residues where at least one heavy atom is within this radius of a heavy
        ligand atom, are considered part of the binding site.
    outlier_distance : float, optional
        The binding sites of the reference and pose are superimposed iteratively.
        In each iteration, atoms with a distance of more than this value are considered
        outliers and are removed in the next iteration.
        To disable outlier removal, set this value to ``inf``.
    max_iterations : int, optional
        The maximum number of iterations for the superimposition.
    min_anchors : int, optional
        The minimum number of anchors to use for the superimposition.
        If less than this number of anchors are present, the superimposition is
        performed on all interface backbone atoms.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.26601
    """

    def __init__(
        self,
        threshold: float,
        inclusion_radius: float = 4.0,
        outlier_distance: float = 3.0,
        max_iterations: int = 5,
        min_anchors: int = 3,
    ) -> None:
        self._threshold = threshold
        self._inclusion_radius = inclusion_radius
        self._outlier_distance = outlier_distance
        self._max_iterations = max_iterations
        self._min_anchors = min_anchors
        super().__init__()

    @property
    def name(self) -> str:
        return "BiSyRMSD"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [(f"<{self._threshold}", 0), (f">{self._threshold}", self._threshold)]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        return bisy_rmsd(
            reference,
            pose,
            inclusion_radius=self._inclusion_radius,
            outlier_distance=self._outlier_distance,
            max_iterations=self._max_iterations,
            min_anchors=self._min_anchors,
        )

    def smaller_is_better(self) -> bool:
        return True


class BondLengthViolations(Metric):
    """
    Check for unusual bond lengths in the structure by comparing against reference
    values.
    Returns the percentage of bonds that are outside acceptable ranges.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance in Angstroms for acceptable deviation from ideal bond lengths.
        Default is 0.1 Angstroms.
    reference_bonds : dict, optional
        Dictionary mapping atom type pairs to ideal bond lengths.
        If not provided, uses a default set of common bond lengths.
    """

    # Default reference bond lengths in Angstroms
    _DEFAULT_REFERENCE_BONDS = {
        ("C", "C"): np.array([1.54, 1.33, 1.20]),
        ("C", "N"): np.array([1.47, 1.27, 1.15]),
        ("C", "O"): np.array([1.43, 1.21]),
        ("N", "H"): np.array([1.01]),
        ("O", "H"): np.array([0.96]),
        ("C", "H"): np.array([1.09]),
        ("C", "S"): np.array([1.82]),
        ("S", "S"): np.array([2.05]),
    }

    def __init__(
        self,
        tolerance: float = 0.1,
        reference_bonds: Dict[Tuple[str, str], np.ndarray] | None = None,
    ) -> None:
        self._tolerance = tolerance
        self._reference_bonds = (
            reference_bonds
            if reference_bonds is not None
            else self._DEFAULT_REFERENCE_BONDS
        )
        # Add reverse pairs for convenience
        reverse_bonds = {(b, a): v for (a, b), v in self._reference_bonds.items()}
        self._reference_bonds.update(reverse_bonds)
        super().__init__()

    @property
    def name(self) -> str:
        return "Bond-length-violation"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [
                ("poor", 0.0),
                ("acceptable", 0.9),
                ("good", 0.95),
                ("excellent", 0.99),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Calculate the percentage of bonds that are outside acceptable ranges.

        Parameters
        ----------
        reference : AtomArray
            Not used in this metric as we compare against ideal bond lengths.
        pose : AtomArray
            The structure to evaluate.

        Returns
        -------
        float
            Percentage of bonds outside acceptable ranges (0.0 to 1.0).
        """
        if pose.bonds is None:
            return np.nan

        total_checked = 0
        valid_bonds = 0

        # Get all bonds and iterate through the bond tuples
        for i, j, _ in pose.bonds.as_array():
            atom1_type = pose.element[i]
            atom2_type = pose.element[j]

            if (atom1_type, atom2_type) in self._reference_bonds:
                bond_length = struc.distance(pose[i], pose[j])
                ideal_lengths = self._reference_bonds.get((atom1_type, atom2_type))
                if ideal_lengths is None:
                    warnings.warn(
                        f"No reference bond length for {atom1_type}-{atom2_type} found. Ignoring {atom1_type}-{atom2_type} bond."
                    )
                    continue
                if np.any(np.abs(bond_length - ideal_lengths) <= self._tolerance):
                    valid_bonds += 1
                total_checked += 1

        if total_checked == 0:
            return np.nan

        return 1 - (valid_bonds / total_checked)

    def smaller_is_better(self) -> bool:
        return True


class BondAngleViolations(Metric):
    """
    Check for unusual bond angles in the structure by comparing against
    idealized bond geometry.
    Returns the percentage of bonds that are outside acceptable ranges.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance in radians for acceptable deviation from ideal bond angles.
    """

    def __init__(self, tolerance: float = np.deg2rad(15.0)) -> None:
        self._tolerance = tolerance
        super().__init__()

    @property
    def name(self) -> str:
        return "Bond-angle-violation"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [
                ("poor", 0.0),
                ("acceptable", 0.9),
                ("good", 0.95),
                ("excellent", 0.99),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Calculate the percentage of bonds that are outside acceptable ranges.

        Parameters
        ----------
        reference : AtomArray
            Not used in this metric as we compare against ideal bond angles.
        pose : AtomArray
            The structure to evaluate.

        Returns
        -------
        float
            Percentage of bonds outside acceptable ranges (0.0 to 1.0).
        """
        if pose.bonds is None:
            return np.nan

        # Idealize the pose local geometry to make the reference
        reference = idealize_bonds(pose)

        # Check the angle of all bonded triples
        graph = reference.bonds.as_graph()
        bonded_triples = np.array(graph_to_connected_triples(graph))
        ref_angles = struc.index_angle(reference, bonded_triples)
        pose_angles = struc.index_angle(pose, bonded_triples)
        angle_diffs = np.abs(ref_angles - pose_angles)
        valid_angles = (angle_diffs <= self._tolerance).sum()
        total_checked = len(bonded_triples)

        if total_checked == 0:
            return np.nan

        return 1 - (valid_angles / total_checked)

    def smaller_is_better(self) -> bool:
        return True


class ClashCount(Metric):
    """
    Count the number of clashes between atoms in the pose.
    """

    @property
    def name(self) -> str:
        return "Number of clashes"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        if pose.array_length() == 0:
            return np.nan
        return len(find_clashes(pose))

    def smaller_is_better(self) -> bool:
        return True


def _run_for_each_monomer(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[[struc.AtomArray, struc.AtomArray], float],
) -> float:
    """
    Run the given function for each monomer in the reference and pose.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable
        The function to run for each monomer.
        Takes the reference and pose as input and returns a scalar value.

    Returns
    -------
    metrics : float
        The average return value of `function`, weighted by the number of atoms.
        If the input structure contains no chains, *NaN* is returned.
    """
    values = []
    chain_starts = struc.get_chain_starts(reference, add_exclusive_stop=True)
    for start_i, stop_i in itertools.pairwise(chain_starts):
        reference_chain = reference[start_i:stop_i]
        pose_chain = pose[start_i:stop_i]
        values.append(function(reference_chain, pose_chain))
    values = np.array(values)

    if len(values) == 0:
        # No chains in the structure
        return np.nan
    else:
        n_atoms_per_chain = np.diff(chain_starts)
        # Ignore chains where the values are NaN
        not_nan_mask = np.isfinite(values)
        return np.average(
            values[not_nan_mask], weights=n_atoms_per_chain[not_nan_mask]
        ).item()


def _run_for_each_chain_pair(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[
        [struc.AtomArray, struc.AtomArray, struc.AtomArray, struc.AtomArray], float
    ],
) -> float:
    """
    Run the given function for each chain pair combination in the reference and pose
    and return the average value.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable[(reference_chain1, reference_chain2, pose_chain1, pose_chain2), float]
        The function to run for each interface.
        Takes the reference and pose chains in contact as input and returns a scalar
        value.
        The function may also return *NaN*, if its result should be ignored.

    Returns
    -------
    metrics : float
        The average return value of `function`, weighted by the number of atoms.
        If the input structure contains only one chain, *NaN* is returned.
    """
    chain_starts = struc.get_chain_starts(reference, add_exclusive_stop=True)
    reference_chains = [
        reference[start:stop] for start, stop in itertools.pairwise(chain_starts)
    ]
    pose_chains = [pose[start:stop] for start, stop in itertools.pairwise(chain_starts)]
    results = []
    for chain_i, chain_j in itertools.combinations(range(len(reference_chains)), 2):
        results.append(
            function(
                reference_chains[chain_i],
                reference_chains[chain_j],
                pose_chains[chain_i],
                pose_chains[chain_j],
            )
        )
    if len(results) == 0:
        return np.nan
    return np.nanmean(results).item()


def _average_over_ligands(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[[struc.AtomArray, struc.AtomArray], float],
) -> float:
    """
    Run the given function for each ligand in the reference and pose.
    and return the average value.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable[[reference_ligand, pose_ligand], float]
        The function to run for each ligand.
        Takes the reference and pose ligand as input and returns a scalar
        value.

    Returns
    -------
    metrics : float
        The average return value of `function`, weighted by the number of atoms.
        If the input structure contains no ligand atoms, *NaN* is returned.
    """
    values = _run_for_each_ligand(reference, pose, function)
    if len(values) == 0:
        # No ligands in the structure
        return np.nan
    else:
        return np.mean(values).item()


def _run_for_each_ligand(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[[struc.AtomArray, struc.AtomArray], Any],
) -> list[Any]:
    """
    Run the given function for each isolated ligand in complex with all proteins from
    the system.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable[[reference_ligand, pose_ligand], float]
        The function to run.
        Takes the reference and pose system as input and returns a scalar
        value.

    Returns
    -------
    metrics : float
        The average return value of `function`.
        If the input structure contains no ligand atoms, *NaN* is returned.
    """
    values = []
    ligand_mask = reference.hetero
    polymer_mask = ~ligand_mask
    chain_starts = struc.get_chain_starts(reference)
    if len(chain_starts) == 0:
        # No chains in the structure
        return []
    chain_masks = struc.get_chain_masks(reference, chain_starts)
    # Only keep chain masks that correspond to ligand chains
    ligand_masks = chain_masks[(chain_masks & ligand_mask).any(axis=-1)]
    for ligand_mask in ligand_masks:
        # Evaluate each isolated ligand in complex separately
        complex_mask = ligand_mask | polymer_mask
        values.append(function(reference[complex_mask], pose[complex_mask]))
    return values


def _select_receptor_and_ligand(
    reference_chain1: struc.AtomArray,
    reference_chain2: struc.AtomArray,
    pose_chain1: struc.AtomArray,
    pose_chain2: struc.AtomArray,
) -> tuple[struc.AtomArray, struc.AtomArray, struc.AtomArray, struc.AtomArray]:
    """
    Select the receptor and ligand for the given interface.

    The longer of both chains is the receptor.

    Parameters
    ----------
    reference_chain1, reference_chain2 : AtomArray, shape=(n,)
        The reference chains.
    pose_chain1, pose_chain2 : AtomArray, shape=(n,)
        The pose chains.

    Returns
    -------
    reference_receptor, reference_ligand, pose_receptor, pose_ligand : AtomArray
        The selected receptor and ligand.
    """
    if is_small_molecule(reference_chain1):
        return (reference_chain2, reference_chain1, pose_chain2, pose_chain1)
    elif is_small_molecule(reference_chain2):
        return (reference_chain1, reference_chain2, pose_chain1, pose_chain2)
    elif reference_chain1.array_length() >= reference_chain2.array_length():
        return (reference_chain1, reference_chain2, pose_chain1, pose_chain2)
    else:
        return (reference_chain2, reference_chain1, pose_chain2, pose_chain1)
