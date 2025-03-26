
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
            :class: only-light, no-scaled-link
            :align: center
            :width: 75%
            :alt: pepp'r

        .. image:: /static/assets/general/logo_dark.svg
            :class: only-dark, no-scaled-link
            :align: center
            :width: 75%
            :alt: pepp'r

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

        |

        Powered by

        .. image:: /static/assets/general/sponsor.svg
            :class: only-light
            :width: 15%
            :alt: VantAI
            :target: https://www.vant.ai/

        .. image:: /static/assets/general/sponsor_dark.svg
            :class: only-dark
            :width: 15%
            :alt: VantAI
            :target: https://www.vant.ai/

    .. grid-item::

        .. grid:: 1

            .. grid-item::

                .. image:: /showcase/system.png
                    :class: only-light, no-scaled-link
                    :alt: Evaluated system
                    :width: 100%
                    :align: center

                .. image:: /showcase/system.png
                    :class: only-dark, no-scaled-link
                    :alt: Evaluated system
                    :width: 100%
                    :align: center

            .. grid-item::

                .. image:: /showcase/metrics.png
                    :class: only-light, no-scaled-link
                    :alt: Metrics
                    :width: 100%
                    :align: center

                .. image:: /showcase/metrics_dark.png
                    :class: only-dark, no-scaled-link
                    :alt: Metrics
                    :width: 100%
                    :align: center

.. toctree::
    :maxdepth: 1
    :hidden:

    tutorial/index
    api
    contribution