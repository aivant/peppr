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
    ClashCount

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

Atom Matching
-------------
.. autosummary::
    :toctree: apidoc
    :caption: Atom Matching

    find_optimal_match
    find_all_matches
    GraphMatchWarning
    UnmappableEntityError

Miscellaneous
-------------

.. autosummary::
    :toctree: apidoc
    :caption: Miscellaneous

    sanitize
    standardize
    is_small_molecule
    get_contact_residues
    MatchWarning
    EvaluationWarning
    NoContactError