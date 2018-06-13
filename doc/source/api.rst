
.. currentmodule: econtools

.. _api:

*******************
Function Signatures
*******************

.. contents::
    :local:
    :depth: 4

Data I/O
~~~~~~~~

.. autofunction:: econtools.load_or_build
.. autofunction:: econtools.save_cli
.. autofunction:: econtools.confirmer
.. autofunction:: econtools.read
.. autofunction:: econtools.write

Econometrics
~~~~~~~~~~~~

.. autofunction:: econtools.metrics.reg
.. autofunction:: econtools.metrics.ivreg
.. autoclass:: econtools.metrics.core.Results
.. automethod:: econtools.metrics.core.Results.Ftest
.. autofunction:: econtools.metrics.f_test
.. autofunction:: econtools.metrics.kdensity
.. autofunction:: econtools.metrics.llr


LaTeX
~~~~~

.. autofunction:: econtools.outreg
.. autofunction:: econtools.table_mainrow
.. autofunction:: econtools.table_statrow
.. autofunction:: econtools.write_notes


Plotting
~~~~~~~~

.. autofunction:: econtools.binscatter
.. autofunction:: econtools.legend_below


Reference
~~~~~~~~~

.. autofunction:: econtools.state_name_to_fips
.. autofunction:: econtools.state_fips_to_name
.. autofunction:: econtools.state_name_to_abbr
.. autofunction:: econtools.state_abbr_to_name
