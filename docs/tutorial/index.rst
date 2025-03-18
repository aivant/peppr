:sd_hide_title: true

########
Tutorial
########

Getting started
===============
``peppr`` is a *Python* package for evaluation of molecular structure poses,
created by any structure prediction tool, against corresponding reference structures,
that represent the perfect *ground truth* for that structure.

Installation
------------
`peppr` can be installed from `PyPI <https://pypi.org/project/peppr/>`_.

.. code-block:: console

    $ pip install peppr

If the installation suceeded you should be able to import `peppr` in *Python*

.. code-block:: python

    import peppr

and to use the command line interface

.. code-block:: console

    $ peppr --help

Usage
-----
As implied above, ``peppr`` can either be used as a command line program or, especially
if higher flexibility is required, as a Python package.
The following tutorials will illuminate both ways and also show how to extend
``peppr`` with custom metrics.

.. grid:: 3

    .. grid-item-card::
        :link: cli
        :link-type: doc
        :text-align: center

        .. grid:: 2

            .. grid-item::
                :columns: 2
                :class: tutorial-button

                .. raw:: html

                    <svg class="tutorial-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M9.4 86.6C-3.1 74.1-3.1 53.9 9.4 41.4s32.8-12.5 45.3 0l192 192c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L178.7 256 9.4 86.6zM256 416l288 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-288 0c-17.7 0-32-14.3-32-32s14.3-32 32-32z"/></svg>

            .. grid-item::
                :columns: 10

                Command line

    .. grid-item-card::
        :link: api
        :link-type: doc
        :text-align: center

        .. grid:: 2

            .. grid-item::
                :columns: 2
                :class: tutorial-button

                .. raw:: html

                    <svg class="tutorial-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M392.8 1.2c-17-4.9-34.7 5-39.6 22l-128 448c-4.9 17 5 34.7 22 39.6s34.7-5 39.6-22l128-448c4.9-17-5-34.7-22-39.6zm80.6 120.1c-12.5 12.5-12.5 32.8 0 45.3L562.7 256l-89.4 89.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l112-112c12.5-12.5 12.5-32.8 0-45.3l-112-112c-12.5-12.5-32.8-12.5-45.3 0zm-306.7 0c-12.5-12.5-32.8-12.5-45.3 0l-112 112c-12.5 12.5-12.5 32.8 0 45.3l112 112c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256l89.4-89.4c12.5-12.5 12.5-32.8 0-45.3z"/></svg>

            .. grid-item::
                :columns: 10

                Python API

    .. grid-item-card::
        :link: custom
        :link-type: doc
        :text-align: center

        .. grid:: 2

            .. grid-item::
                :columns: 2
                :class: tutorial-button

                .. raw:: html

                    <svg class="tutorial-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M208 80c0-26.5 21.5-48 48-48l64 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-8 0 0 40 152 0c30.9 0 56 25.1 56 56l0 32 8 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-64 0c-26.5 0-48-21.5-48-48l0-64c0-26.5 21.5-48 48-48l8 0 0-32c0-4.4-3.6-8-8-8l-152 0 0 40 8 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-64 0c-26.5 0-48-21.5-48-48l0-64c0-26.5 21.5-48 48-48l8 0 0-40-152 0c-4.4 0-8 3.6-8 8l0 32 8 0c26.5 0 48 21.5 48 48l0 64c0 26.5-21.5 48-48 48l-64 0c-26.5 0-48-21.5-48-48l0-64c0-26.5 21.5-48 48-48l8 0 0-32c0-30.9 25.1-56 56-56l152 0 0-40-8 0c-26.5 0-48-21.5-48-48l0-64z"/></svg>

            .. grid-item::
                :columns: 10

                Customization

.. toctree::
    :hidden:
    :maxdepth: 1

    cli
    api
    custom