.. _overview:

.. currentmodule: econtools

.. toctree::
    :hidden:

    io.rst
    metrics_advanced.rst
    to_latex.rst
    plot.rst
    reference.rst


****************
Package overview
****************

:mod:`econtools` consists of several sets of tools commonly used for
econometrics and data manipulation.

.. contents:: :local:

Input/Output
------------

:mod:`econtools` contains a number of boilerplate methods that make it easier to
create datasets, save them to disk, and prepare them for statistical analysis.

Highlights include

* :py:func:`~econtools.util.io.load_or_build`: a decorator that caches a DataFrame to disk.
* :py:func:`~econtools.util.io.save_cli`: an ``argparse`` wrapper that
  adds a ``--save`` command line switch to any script.
* :py:func:`~econtools.confirmer`: a drop-in interactive method that prompts the
  user for a yes or no response, e.g. "Are you sure you want to delete all your
  data?"

Econometric Tools
-----------------

The econometrics tools in :mod:`econtools` include

* Common regression techniques (OLS, 2SLS, LIML)
* Option to absorb any variable via within transformation. This is similar to
  the ``areg, absorb`` command in Stata but it can be used with any relevant
  regression command. This consolidates ``reg``, ``areg``, ``xtreg``, etc., and
  uses the correct degrees of freedom correction.
* Robust standard errors

  * HAC (robust/HC1, HC2, HC3)
  * Clustered standard errors
  * Spatial HAC (aka SHAC or Conley standard errors) with uniform or triangle
    kernel

* F-tests by variable name or arbitrary ``R`` matrix.
* Kernel density estimation
* Local linear regression


LaTeX Tools
-----------


Plotting Tools
--------------


Reference Tools
---------------
