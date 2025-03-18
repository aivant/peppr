:sd_hide_title: true

============
Contribution
============

Contributor guide
=================
If you find ``peppr`` helpful, but still see room for improvement or features to be
added, we welcome you to contribute to the project!
The `Github repository <https://github.com/aivant/peppr>`_ is the central point where
all development and discussion takes place.
This document aims to introduce you to the general development guidelines.

Environment
-----------
For the complete development workflow, additional tools are required as described in
the next sections.
To install ``peppr`` in editable mode together with all of these extra dependencies,
run

.. code-block:: console

   $ pip install -e ".[lint,tests,docs]"

Code style
----------
``peppr`` is compliant with :pep:`8` and uses `Ruff <https://docs.astral.sh/ruff/>`_ for
code formatting and linting.
The maximum line length is 88 characters.
An exception is made for docstring lines, if it is not possible to use a
maximum of 88 characters (e.g. tables and parameter type descriptions).
To make code changes ready for a pull request, simply run

.. code-block:: console

   $ ruff format
   $ ruff check --fix

and fix the remaining linter complaints.

Testing
-------
``peppr`` uses `Pytest <https://docs.pytest.org>`_ to test its code base.
Tests are located in the ``tests/`` directory and are run with

.. code-block:: console

   $ pytest

Documentation
-------------
The documentation is written in
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
and built with `Sphinx <https://www.sphinx-doc.org>`_.

The documentation is built with

.. code-block:: console

    $ sphinx-build docs build/docs

In addition to the ``.rst`` documents in the ``docs/`` directory, the most important
information are the docstrings of each public function and class.
``peppr`` uses the
`numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for all of
its docstrings.
To validate correct formatting, run

.. code-block:: console

   $ numpydoc lint src/peppr/**/*.py

If you add a new function or class, do not forget to add it to the corresponding
``autosummary`` section in ``docs/api.rst``.