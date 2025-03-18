| ``peppr`` is a package for evaluation of predicted poses, right?

Yes, indeed!
It supports computing all *CASP*/*CAPRI* metrics and more on your structure
predictions.
It is even easily extensible to add custom metrics, the possibilities are (almost)
unlimited!
Either use the command line interface or the Python API based on ``AtomArray`` objects
from the `Biotite <https://www.biotite-python.org>`_ library.

Installation
------------

``peppr`` is available via *PyPI*:

.. code-block:: console

    $ pip install peppr

Usage example
-------------

.. code-block:: python

    import peppr

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