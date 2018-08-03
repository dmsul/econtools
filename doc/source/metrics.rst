
.. currentmodule: econtools

.. _metrics:

****************************
Econometrics Tools
****************************

.. contents:: :local:

OLS
---

To estimate an OLS regression, you pass the :py:func:`~econtools.metrics.reg`
function at least three arguments

#. The DataFrame that contains the data.
#. The name of the dependent variable as a string.
#. The name(s) of the independent variable(s) as a string (for one variable) or
   as a list.

Following these arguments, there are a number of keyword arguments for various
other options. For example, the following code estimates a basic wage
regression with state-level clustering and fixed effects, weighting by the
variable ``sample_wt``.

.. code-block:: python

    import pandas as pd
    import econtools.metrics as mt

    # Load a data file with columns 'ln_wage', 'educ', and 'state'
    df = pd.read_csv('my_data.csv')

    y = 'wage'
    X = ['educ', 'age', 'male']
    fe_var = 'state'
    cluster_var = 'state'
    weights_var = 'sample_wt'

    results = mt.reg(
        df,                     # DataFrame
        y,                      # Dependent var (string)
        X,                      # Independent var(s) (string or list of strings)
        a_name=fe_var,          # Fixed-effects/absorb var (string)
        cluster=cluster_var     # Cluster var (string)
        awt_name=weights_var    # Sample weights
    )


Note that :py:func:`~econtools.metrics.reg` does *not* automatically estimate a
constant term. In order to have a constant/intercept in your model, you can (a)
add a column of ones to your DataFrame, or (b) use the ``addcons`` keyword arg:

.. code-block:: python

    results = mt.reg(
        df,
        y,
        X,              # does not include a constant/intercept
        addcons=True    # Adds a constant term
    )


Instrumental Variables
----------------------

Estimating an instrumental variables model is very similar, but is done using
the :py:func:`~econtools.metrics.ivreg` function. The order of arguments is
also slightly different in order to differentiate between the instruments,
endogenous regressors, and exogenous regressors. Other keyword options, such as
``addcons``, ``cluster``, and so forth, are exactly the same as with
:py:func:`~econtools.metrics.reg`.

One additional keyword argument is `method`, which sets the IV method used to
estimate the model. Currently supported values are `2sls` (the default) and
`liml`.

.. code-block:: python

    # <Imports and loading data>

    y = 'wage'              # Dependent var
    X = ['educ']            # Endogenous regressor(s)
    Z = ['treatment']       # Instrumental variable(s)
    W = [ 'age', 'male']    # Exogenous regressor(s)

    results = mt.ivreg(df, y, X, Z, W)


Returned Results
----------------

The regression functions :py:func:`~econtools.metrics.reg` and
:py:func:`~econtools.metrics.ivreg` return a custom
:py:class:`~econtools.metrics.core.Results` object that contains beta
estimates, variance-covariance matrix, and other relevant info.

The easiest way to see regression results is the ``summary`` attribute. But
direct access to estimates is also possible.

.. code-block:: python

    import pandas as pd
    import econtools.metrics as mt

    df = pd.read_stata('some_data.dta')
    results = mt.reg(df, 'ln_wage', ['educ', 'age'], addcons=True)

    # Print DataFrame w/ betas, se's, t-stats, etc.
    print(results.summary)

    # Print only betas
    print(results.beta)

    # Print std. err. for `educ` coefficient
    print(results.se['educ'])

    # Print full variance-covariance matrix
    print(results.vce)


The full list of attributes is listed :py:class:`here <econtools.metrics.core.Results>`.

F tests
~~~~~~~

:py:mod:`econtools.metrics` contains two functions for conducting F tests.

The first, :py:meth:`~econtools.metrics.core.Results.Ftest`, is for simple,
Stata-like tests for joint significance or equality. It is a method on the
:py:class:`~econtools.metrics.core.Results` object.

.. code-block:: python

    results = mt.reg(df, 'ln_wage', ['educ', 'age'], addcons=True)

    # Test for joint significance
    F1, pF1 = results.Ftest(['educ', 'age'])
    # Test for equality
    F2, pF2 = results.Ftest(['educ', 'age'], equal=True)

The second, :py:func:`~econtools.metrics.f_test`, is for F tests of arbitrary
linear combinations of coefficients. The tests are defined by an ``R``
matrix and an ``r`` vector such that the null hypothesis is :math:`R\beta = r`.


Spatial HAC (Conley errors)
---------------------------

Spatial HAC standard errors (as in
`Conley (1999)
<https://www.sciencedirect.com/science/article/pii/S0304407698000840>`_, 
`Kelejian and Prucha (2007)
<https://www.sciencedirect.com/science/article/pii/S0304407606002260>`_,
etc.) can be calculated by passing a dictionary with the relevant fields to the
``shac`` keyword:

.. code-block:: python

    shac_params = {
        'x': 'longitude',   # Column in `df`
        'y': 'latitude',    # Column in `df`
        'kern': 'unif',     # Kernel name
        'band': 2,          # Kernel bandwidth
    }
    df = pd.read_stata('reg_data.dta')
    results = mt.reg(df, 'lnp', ['sqft', 'rooms'],
                     a_name='state',
                     shac=shac_params)


.. Important::

    The ``band`` parameter is assumed to be in the same units as ``x`` and
    ``y``. If ``x`` and ``y`` are degrees latitude/longitude, ``band`` should
    also be in degrees. ``econtools`` does not do any advanced geographic
    distance calculations here, just simple euclidean distance.


Local Linear Regression
-----------------------

See :py:func:`~econtools.metrics.llr`.
