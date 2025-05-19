Custom metrics and selectors
============================

.. jupyter-execute::
    :hide-code:

    from pathlib import Path
    import os
    import pandas as pd

    path_to_systems = Path(os.getcwd()) / "tests" / "data" / "predictions"
    pd.options.display.float_format = lambda x: "{:.2f}".format(x)

.. currentmodule:: peppr

``peppr`` is written with extensibility in mind:
You can easily define your own :class:`Metric` and :class:`Selector` classes
to extend the functionality of the package.

Creating a custom metric
------------------------
To create a custom metric, one only needs to subclass the :class:`Metric` base class.
For the scope of this tutorial we will create an all-atom RMSD as metric.

.. jupyter-execute::

    import peppr
    import numpy as np
    import biotite.structure as struc

    class RMSD(peppr.Metric):

        # This is mandatory
        @property
        def _default_name(self):
            return "My awesome RMSD"

        # This optionally enables binning for `Evaluator.summarize_metrics()`
        @property
        def thresholds(self):
            return {"good": 0.0, "ok": 10.0, "bad": 20.0}

        # The core of each metric
        def evaluate(self, reference, pose):
            # It is not guaranteed that `reference` and `pose` are superimposed
            pose, _ = struc.superimpose(reference, pose)
            return struc.rmsd(reference, pose).item()

        # Defines whether a smaller or larger value indicates a better prediction
        # Used by the `Selector` classes
        def smaller_is_better(self):
            return True

A few notes on the :meth:`evaluate()` method:
The purpose of a :class:`Metric` is to be used in context of an :class:`Evaluator`.
As the :class:`Evaluator` already automatically preprocesses the input systems passed
to :meth:`Evaluator.feed()`, we can rely on some properties of each system that make
our life easier:

- The input system will only contain heavy atoms (i.e. no hydrogen).
- The input system always contains matching atoms.
  This means that atom ``reference[i]`` corresponds to the atom ``pose[i]``.
  Finding the optimal atom mapping in case of ambiguous scenarios, as they appear in
  homomers or symmetric molecules, is already handled under the hood beforehand.
- The :class:`Metric` should respect the ``hetero`` annotation of the input
  :class:`biotite.structure.AtomArray`.
  An atom where ``hetero`` is ``False`` is defined as polymer and where ``hetero`` is
  ``True`` as small molecule.
  For example a peptide chain with ``hetero`` being ``True`` should still be interpreted
  as a small molecule.
- Atoms with the same ``chain_id`` should be considered as part of the same molecule,
  atoms with different ``chain_id`` should be considered as different molecules.

Creating a custom selector
--------------------------
Custom selectors can also be created by inheriting from the :class:`Selector` base
class.
Let's create a selector that picks the *worst case* prediction, i.e. the pose
with the worst value for a given metric.

.. jupyter-execute::

    class WorstCaseSelector(peppr.Selector):

        @property
        def name(self):
            return "worst case"

        def select(self, values, smaller_is_better):
            if smaller_is_better:
                return np.nanmax(values)
            else:
                return np.nanmin(values)

Note that the :meth:`Selector.select()` method does not know anything about the actual
pose or metric it is selecting from.
It only obtains values to select from (may also contain *NaN* values) and their
ordering.

Bringing it all together
------------------------
Now that we have created our custom metric and selector, we can use them in the
:class:`Evaluator` to evaluate our predictions.

.. jupyter-execute::

    import biotite.structure.io.pdbx as pdbx

    evaluator = peppr.Evaluator([RMSD()])

    for system_dir in sorted(path_to_systems.iterdir()):
        if not system_dir.is_dir():
            continue
        system_id = system_dir.name
        pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
        reference = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
        poses = []
        for pose_path in sorted(system_dir.glob("poses/*.cif")):
            pdbx_file = pdbx.CIFFile.read(pose_path)
            poses.append(pdbx.get_structure(pdbx_file, model=1, include_bonds=True))
        evaluator.feed(system_id, reference, poses)

    evaluator.tabulate_metrics(selectors=[WorstCaseSelector()])

And finally, we will also the bin thresholds for the RMSD metric in action.

.. jupyter-execute::

    for name, value in evaluator.summarize_metrics(
        selectors=[WorstCaseSelector()]
    ).items():
        print(f"{name}: {value:.2f}")
