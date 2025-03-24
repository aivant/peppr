It's a package for evaluation of predicted poses, right?

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

.. code-block:: console

    $ peppr run dockq reference.cif poses.cif

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