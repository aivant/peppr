pepp'r
======

    I have a structure prediction model and now I want to know how well it performs in
    reproducing the reference structures.
    But there are so many possible metrics, some for monomers, some for complexes!
    Is there a package that handles this for me?

|

Try

.. image:: https://raw.githubusercontent.com/aivant/peppr/refs/heads/main/docs/static/assets/general/logo.svg
   :alt: pepp'r

|

    It's a Package for Evaluation of Predicted Poses, Right?

|

Yes, indeed!
It allows you to compute a variety of metrics on your structure predictions
for assessing their quality.
It supports

- all *CASP*/*CAPRI* metrics and more
- small molecules to huge protein complexes
- easy extension with custom metrics
- a command line interface and a Python API

Installation
------------

``peppr`` is available via *PyPI*:

.. code-block:: console

    $ pip install peppr

Usage example
-------------

Using the CLI, you can either compute a single metric for a system...

.. code-block:: console

    $ peppr run dockq reference.cif poses.cif

... or run an entire prediction model evaluation on many systems.

.. code-block:: console

    # Select the metrics you want to compute (here: RMSD and lDDT)
    $ peppr create peppr.pkl monomer-rmsd monomer-lddt

    # Run the evaluation on predicted poses and their corresponding references
    $ peppr evaluate-batch peppr.pkl "systems/*/reference.cif" "systems/*/poses"

    # Select the aggregation method over poses (here: Top-3 and Oracle) and report the results
    $ peppr tabulate peppr.pkl table.csv top3 oracle

Available metrics
-----------------

- RMSD
- TM-score
- lDDT
- lDDT-PLI
- fnat
- iRMSD
- LRMSD
- DockQ

... and more!