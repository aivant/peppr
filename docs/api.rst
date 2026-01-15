:sd_hide_title: true

==========
Python API
==========

API reference
=============

.. currentmodule:: peppr

.. autoclass:: Evaluator
    :members:

Metrics
-------
The metrics for pose evaluation.

.. autosummary::
    :template: detailed_class.rst
    :toctree: apidoc
    :caption: Metrics

    Metric
    MonomerRMSD
    MonomerTMScore
    MonomerLDDTScore
    IntraLigandLDDTScore
    LDDTPLIScore
    LDDTPPIScore
    GlobalLDDTScore
    DockQScore
    LigandRMSD
    InterfaceRMSD
    ContactFraction
    PocketAlignedLigandRMSD
    BiSyRMSD
    BondLengthViolations
    BondAngleViolations
    ChiralityViolations
    ClashCount
    PLIFRecovery
    PocketDistance
    PocketVolumeOverlap
    RotamerViolations
    RamachandranViolations
    LigandValenceViolations

Selectors
---------
Selection of the desired metric result from multiple poses.

.. autosummary::
    :template: detailed_class.rst
    :toctree: apidoc
    :caption: Selectors

    Selector
    MeanSelector
    MedianSelector
    OracleSelector
    TopSelector
    RandomSelector

Analysis functions
------------------
Underlying functions used be the :class:`Metric` classes to compute the metric values,
that are not directly implemented :mod:`biotite.structure`.

.. autosummary::
    :toctree: apidoc
    :caption: Analysis functions

    bisy_rmsd
    find_clashes
    dockq
    pocket_aligned_lrmsd
    lrmsd
    irmsd
    fnat
    DockQ
    ContactMeasurement
    volume
    volume_overlap
    RotamerScore
    RamaScore
    get_fraction_of_rotamer_outliers
    get_fraction_of_rama_outliers

Atom Matching
-------------
.. autosummary::
    :toctree: apidoc
    :caption: Atom Matching

    find_optimal_match
    find_all_matches
    find_matching_centroids
    filter_matched
    GraphMatchWarning
    UnmappableEntityError
    StructureMismatchError

Miscellaneous
-------------
.. autosummary::
    :toctree: apidoc
    :caption: Miscellaneous

    MoleculeType
    sanitize
    standardize
    get_contact_residues
    find_atoms_by_pattern
    estimate_formal_charges
    find_resonance_charges
    MatchWarning
    EvaluationWarning
    NoContactError
