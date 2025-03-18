
:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

.. raw:: html

    <style>
        .bd-main .bd-content .bd-article-container {
            /* Wider page */
            max-width: 80rem;
        }
    </style>


####################
pepp'r documentation
####################

.. grid:: 1 1 2 2

    .. grid-item::

        .. image:: /static/assets/general/logo.svg
            :alt: pepp'r
            :class: no-scaled-link
            :width: 75%
            :align: center

        .. raw:: html

            <h3>It's a package for evaluation of predicted poses, right?</h2>

        Yes, indeed!
        It allows you to compute a variety of metrics on your structure predictions
        for assessing their quality.
        It supports

        - all *CASP*/*CAPRI* metrics and more
        - small molecules to huge protein complexes
        - easy extension with custom metrics
        - a command line interface and a Python API

        .. grid:: 3

            .. grid-item::

                .. button-ref:: tutorial/index
                    :color: primary
                    :expand:
                    :shadow:

                    Show me how!

    .. grid-item::

        .. grid:: 1

            .. grid-item::

                .. image:: /showcase/system.png
                    :alt: Evaluated system
                    :class: no-scaled-link
                    :width: 80%
                    :align: center

            .. grid-item::

                .. image:: /showcase/metrics.png
                    :alt: Metrics
                    :class: no-scaled-link
                    :width: 100%
                    :align: center

.. toctree::
    :maxdepth: 1
    :hidden:

    tutorial/index
    api
    contribution