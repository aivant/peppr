Python API
==========

.. jupyter-execute::
    :hide-code:

    from pathlib import Path
    import os
    import pandas as pd

    path_to_systems = Path(os.getcwd()) / "tests" / "data" / "predictions"
    pd.options.display.float_format = lambda x: "{:.2f}".format(x)

.. currentmodule:: peppr

This tutorial demonstrates how to use ``peppr`` from within Python,
from loading structures to evaluating and reporting metrics.

Loading and preparing structures
--------------------------------
``peppr`` leverages the structure representation from the
`Biotite <https://www.biotite-python.org>`_ package, namely the
:class:`biotite.structure.AtomArray`.
Consequently, ``peppr`` does not provide a way to load or edit structures by itself:
All this is done 'upstream' via functionalities from Biotite.
Have a look at the
`corresponding Biotite tutorials <https://www.biotite-python.org/latest/tutorial/structure/index.html>`_
for a more in-depth introduction on its capabilities.

For the scope of this tutorial we will load our reference structures and predicted poses
from CIF files.
The files used in this tutorial are *proteolysis targeting chimera* (PROTAC) complexes
predicted from the `PLINDER <https://github.com/plinder-org/plinder>`_ dataset and can
be downloaded :download:`here </download/systems.zip>`.
As input structures for ``peppr`` may comprise any number of peptide and nucleotide
chains, as well as small molecules, they are called *system* throughout this package.

.. jupyter-execute::

    import biotite.structure.io.pdbx as pdbx

    system_dir = path_to_systems / "7jto__1__1.A_1.D__1.J_1.O"
    pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
    ref = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    pdbx_file = pdbx.CIFFile.read(system_dir / "poses" / "pose_0.cif")
    pose = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    print(type(ref))
    print(type(pose))

There are two important things to note here:

- ``model=1`` ensures that only one reference and one pose is loaded from the CIF file,
  as the file may also contain multiple models.
  You can omit it, if you want ``peppr`` to evaluate multiple poses per system, as
  discussed later.
- ``include_bonds=True`` instructs the parser also to load the bonds between atoms.
  If they are missing ``peppr`` will raise an exception when it sees this system.
  The full list of requirements on input systems is documented in the
  :doc:`API reference </api>`.

As already indicated above, :class:`biotite.structure.AtomArray` objects can also be
loaded from a variety of other formats, such as PDB and MOL/SDF.
Take a look at :mod:`biotite.structure.io` for a comprehensive list.

Evaluation of a metric on a structure
-------------------------------------
Now that we have the reference and poses loaded, we can evaluate metrics on them.
Each metric is represented by a :class:`Metric` object.

.. jupyter-execute::

    import peppr

    lddt_metric = peppr.GlobalLDDTScore(backbone_only=False)
    print("Name of the metric:", lddt_metric.name)
    print(
        "Which values are considered better predictions?",
        "Smaller ones." if lddt_metric.smaller_is_better() else "Larger ones."
    )

The :meth:`Metric.evaluate()` method takes the reference and poses of a system
and returns a scalar value.

.. jupyter-execute::

    print(lddt_metric.evaluate(ref, pose))

Note that :meth:`Metric.evaluate()` requires that the reference and pose have matching
atoms, i.e. ``ref[i]`` and ``pose[i]`` should point to corresponding atoms.
If this is not the case, you can use :func:`find_matching_atoms()` to get indices that
bring the atoms into the correct order.

.. jupyter-execute::

    ref_indices, pose_indices = peppr.find_matching_atoms(ref, pose)
    ref = ref[ref_indices]
    pose = pose[pose_indices]

Some metrics may not be defined for a given system.
For example, if we remove the second protein chain from the current system, the
*interface RMSD* (iRMSD) is undefined.
In such cases, :meth:`Metric.evaluate()` returns *NaN*.

.. jupyter-execute::

    # Remove one protein chain
    mask = ref.chain_id != "1"
    ref = ref[mask]
    pose = pose[mask]
    irmsd_metric = peppr.InterfaceRMSD()
    print(irmsd_metric.evaluate(ref, pose))

Feeding the evaluator
---------------------
Up to now we looked at each prediction in isolation.
However, to evaluate a structure prediction model, one usually wants to run evaluations
on many systems and eventually aggregate the results.
This purpose is fulfilled by the :class:`Evaluator` class.
Given the desired metrics, predictions can be fed into it one after the other.
As a bonus, the :class:`Evaluator` also takes care of finding the corresponding atoms
between the reference structure and its poses, in case some atoms are missing in either
one, so you do not have to.

.. jupyter-execute::

    # An evaluator focused on metrics for protein-protein and protein-ligand interactions
    evaluator = peppr.Evaluator(
        [
            peppr.DockQScore(),
            peppr.ContactFraction(),
            peppr.InterfaceRMSD(),
            # Only defined for protein-protein interactions
            peppr.LigandRMSD(),
            # Only defined for protein-ligand interactions
            peppr.PocketAlignedLigandRMSD(),
        ]
    )

    for system_dir in sorted(path_to_systems.iterdir()):
        if not system_dir.is_dir():
            continue
        system_id = system_dir.name
        pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
        ref = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
        pdbx_file = pdbx.CIFFile.read(system_dir / "poses" / "pose_0.cif")
        pose = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
        evaluator.feed(system_id, ref, pose)

Eventually we can report the evaluation run as a table that lists every chosen metric
for each system (indicated by the given system ID).

.. jupyter-execute::

    evaluator.tabulate_metrics()

As mentioned above, you probably also want to aggregate the results over the systems.
This can be done by calling :meth:`Evaluator.summarize_metrics()` which gives a
dictionary mapping the metric names to their aggregated values.

.. jupyter-execute::

    for name, value in evaluator.summarize_metrics().items():
        print(f"{name}: {value:.2f}")

As you see for each metric the mean and median value are reported.
You might also notice that in case of the :class:`DockQScore` also
*incorrect*, *acceptable*, *medium* and *high* percentages are reported.
For example, the value for ``DockQ acceptable`` reads as

| A fraction of ``<x>`` of all applicable systems are within the *acceptable* threshold.

where the thresholds are defined in the ``thresholds`` attribute of some :class:`Metric`
classes such as :class:`DockQScore`.

.. jupyter-execute::

    for bin, lower_threshold in peppr.DockQScore().thresholds.items():
        print(f"{bin}: {lower_threshold:.2f}")

Selecting the right pose
------------------------
Until now we fed only a single pose per system to the :class:`Evaluator`.
However, many structure prediction models return a number of poses, potentially ranked
by their predicted confidence.
To reflect this, the :class:`Evaluator` accepts multiple poses per system - either
via passing a list of :class:`biotite.structure.AtomArray` objects or an
:class:`biotite.structure.AtomArrayStack`, which represents multiple poses with the
same atoms.

.. jupyter-execute::

    system_dir = path_to_systems / "7jto__1__1.A_1.D__1.J_1.O"
    pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
    reference = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    poses = []
    for pose_path in sorted(system_dir.glob("poses/*.cif")):
        pdbx_file = pdbx.CIFFile.read(pose_path)
        poses.append(pdbx.get_structure(pdbx_file, model=1, include_bonds=True))

But for each metric only one value is finally reported, so how does the
:class:`Evaluator` choose a single result for multiple poses?
The answer is, using the current setup the :class:`Evaluator` cannot, you will see
an exception instead.

.. jupyter-execute::
    :raises: ValueError

    evaluator = peppr.Evaluator([peppr.DockQScore()])
    evaluator.feed("foo", reference, poses)
    evaluator.tabulate_metrics()

The :class:`Evaluator` needs a way to select the desired value from the metric evaluated
on each pose.
What is desired depends on the use case: Do we only want the value for the most
confident prediction or the best prediction or the average of all predictions?
Hence, the user needs to supply one or multiple :class:`Selector` objects to the
:meth:`Evaluator.tabulate()` and :meth:`Evaluator.summarize()` methods.
For example the :class:`TopSelector` selects the best value from the `k` most confident
predictions.

.. note::

    :meth:`Evaluator.feed()` expects that poses are sorted from highest to lowest
    confidence, i.e the first pose in the list is the most confident prediction.

.. jupyter-execute::

    evaluator.tabulate_metrics(selectors=[peppr.TopSelector(3)])

Note the added `(Top3)` in the column name:
For every given selector a column will appear that shows the selected value for
that :class:`Selector`.

.. jupyter-execute::

    evaluator.tabulate_metrics(selectors=[peppr.TopSelector(3), peppr.MedianSelector()])