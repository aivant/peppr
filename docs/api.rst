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
    ClashCount

Selectors
---------
Selection of the desired metric result from multiple poses.

.. autosummary::
    :toctree: apidoc
    :caption: Selectors

    Selector
    MeanSelector
    MedianSelector
    OracleSelector
    TopSelector

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

Miscellaneous
-------------

.. autosummary::
    :toctree: apidoc
    :caption: Miscellaneous

    MatchWarning
    GraphMatchWarning
    EvaluationWarning
    NoContactError
    sanitize
    standardize
    find_matching_atoms
    is_small_molecule
    get_contact_residues