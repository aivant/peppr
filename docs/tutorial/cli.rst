Command line interface
======================
One way to use ``peppr`` to compute a single metric or run entire model evaluations is
via the command line interface.
In ``peppr`` we call each input structure, which consists of a reference and one or more
poses, a *system*.
This tutorial uses *proteolysis targeting chimera* (PROTAC) complexes predicted from the
`PLINDER <https://github.com/plinder-org/plinder>`_ dataset as systems to be evaluated
(download :download:`here </download/systems.zip>`) and assumes that they are located in
a directory called ``$SYSTEMDIR``.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os
    os.environ["SYSTEMDIR"] = os.path.join(os.environ["TMPDIR"], "systems")

    ! rm -r $SYSTEMDIR
    ! cp -r tests/data/predictions $SYSTEMDIR

.. jupyter-execute::

    ! ls --color=never $SYSTEMDIR

Each system subdirectory is formatted as follows:

.. jupyter-execute::

    ! cd $SYSTEMDIR && find 7jto__1__1.A_1.D__1.J_1.O

Despite the tutorial using only *CIF* files, ``peppr`` also supports the *BinaryCIF*,
*PDB* and *MOL/SDF* formats.

Quick computation of a single metric
------------------------------------
Sometimes one only wants to know a single metric for a given prediction.
This use case is fulfilled by the ``peppr run`` command.
Let's say we want to know the *DockQ* score for the system
``7jto__1__1.A_1.D__1.J_1.O``.

.. jupyter-execute::

    ! peppr run dockq $SYSTEMDIR/7jto__1__1.A_1.D__1.J_1.O/reference.cif $SYSTEMDIR/7jto__1__1.A_1.D__1.J_1.O/poses/pose_0.cif

To get a list of all available metrics, consult the help output of ``peppr run``.

.. jupyter-execute::

    ! peppr run --help

Running an evaluation on many systems
-------------------------------------
The typical use case is to evaluate a set of metrics of interest on a number of
predicted systems.
To retain a bit of flexibility from the Python API here, an evaluation is split into
three steps revolving around a binary ``peppr.pkl`` file:

1. Create the ``peppr.pkl`` file, defining the metrics that will be computed.
2. Feed the ``peppr.pkl`` file with the systems to be evaluated.
3. Tabulate and/or summarize the results.

.. warning::

    It is not guaranteed that the ``peppr.pkl`` file works across different
    platforms, Python versions and ``peppr`` versions.
    Furthermore, it is not secure to read a ``peppr.pkl`` file from untrusted sources.
    Hence, it is advised to treat it as an ephemeral file and rather use the reported
    summary or table as persistent results.

Evaluator creation
^^^^^^^^^^^^^^^^^^
In the beginning we need to decide which metrics we want to compute for our systems.
For this tutorial we arbitrarily choose the *RMSD* and *lDDT* of each monomer.
The available metrics are actually the same as listed in the help output above.
We create the ``peppr.pkl`` file including these metrics with ``peppr create``.

.. jupyter-execute::

    ! peppr create $TMPDIR/peppr.pkl monomer-rmsd monomer-lddt

Feeding systems to the evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we can start evaluating systems.
There are two possibilities:
Either we feed the systems to the evaluator one by one using ``peppr evaluate``,
or we can feed the systems to the evaluator as a whole using ``peppr evaluate-batch``.
Here we will use the latter.
``peppr evaluate-batch`` expects the references and poses to be provided as
*glob patterns*.
In our case, the systems are spread across the subdirectories of ``$SYSTEMDIR``, so
these directories will be represented by a ``*`` in the glob pattern.
As each system has only one reference, the reference pattern needs to target single
files, while the pose pattern may either point also to single files
(single pose per system) or to directories, where every file represents a pose for
a single system.
The command also needs the path to the previously created ``peppr.pkl`` file, which
will be updated with the results.

.. jupyter-execute::

    ! peppr evaluate-batch $TMPDIR/peppr.pkl "$SYSTEMDIR/*/reference.cif" "$SYSTEMDIR/*/poses"

.. warning::

    Be careful when crafting the *glob patterns*:
    ``peppr`` expands the patterns to paths and sorts them lexicographically.
    Then the ``REFERENCE`` paths are matched to the ``POSE`` paths in this order.
    This means if the directory structure is not named consistently or the
    *glob patterns* are erroneous, the poses may be assigned silently to the wrong
    references, giving wrong results.
    However, this scenario may only occur in edge cases, as commonly erroneous patterns
    lead to a different number of poses and references, which is reported as error.

Note that for multi-pose systems, the lexicographical order of the poses is assumed to
be also the order of confidence.
For example in a directory with ``pose_0.cif, ..., pose_<n>.cif``, ``pose_0.cif`` is
assumed to be the most confident prediction.

Tabulating results
^^^^^^^^^^^^^^^^^^
Finally we can report the results of the evaluation stored in the ``peppr.pkl`` file.
One way is a table listing each metric evaluated for each system with
``peppr tabulate``.
If the systems have multiple poses, we also need to tell ``peppr`` from which pose the
value should be picked.
This is done with selectors.
In this case we select the best value of the three most confident poses.

.. jupyter-execute::

    ! peppr tabulate $TMPDIR/peppr.pkl $TMPDIR/table.csv top3
    ! cat $TMPDIR/table.csv

.. note::

    The term '*polymer*' is used throughout ``peppr`` to refer to protein or nucleic
    acid chains.

Aggregating results over systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The other type of report is an aggregated value for each metric (and each selector)
via ``peppr summarize``.
Again we need to pass at least one selector to ``peppr``, if we have multiple poses per
system.

.. jupyter-execute::

    ! peppr summarize $TMPDIR/peppr.pkl $TMPDIR/summary.json top3
    ! cat $TMPDIR/summary.json

For each metric the mean and median value over all systems are reported.
For some metrics, such as ``rmsd`` and ``dockq``, there is also bins the values are
sorted into.
For example, ``backbone RMSD <2.0`` gives the percentage of systems with a
:math:`C_{\alpha}`-RMSD below 2.0 Å.

Increasing the accuracy
-----------------------
One of the main challenges ``peppr`` solves is the atom matching between the reference
and pose in the first place.
The default method is fast, though in rare cases might miss the mapping that would
optimize the metric of interest.
To employ a more precise but slower method, ``peppr create`` and ``peppr run`` accept
the ``--match-method`` option.

.. jupyter-execute::

    ! peppr create $TMPDIR/peppr.pkl lddt-ppi lddt-pli --match-method exhaustive

For more background, this topic is discussed more deeply in the
:ref:`Python API <More speed or higher accuracy?>` tutorial.