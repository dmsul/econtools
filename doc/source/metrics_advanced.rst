
.. _metricsadv:

.. currentmodule: econtools

****************************
Econometrics: Advanced Usage
****************************

OLS
---

To estimate an OLS regression, you pass the :py:meth:`~econtools.metrics.reg`
function at least three arguments

#. The DataFrame that contains the data.
#. The name of the dependent variable as a string.
#. The name(s) of the independent variable(s) as a string (for one variable) or
   as a list.

Following these arguments, there are a number of keyword arguments for various
other options. For example, the following code estimates a basic wage
regression with state-level clustering and fixed effects, weighting by the
variable `'sample_wt'`.

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


As of now :py:meth:`~econtools.metrics.reg` does *not* automatically estimate a
constant term. In order to have a constant/intercept in your model, you can (a)
add a column of ones to your DataFrame, or (b) use the `addcons` keyword arg:

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
the :py:meth:`~econtools.metrics.ivreg` function. The order of arguments is
also slightly different in order to differentiate between the instruments,
endogenous regressors, and exogenous regressors. Other keyword options, such as
`addcons`, `cluster`, and so forth, are exactly the same as with
:py:meth:`~econtools.metrics.reg`.

One additional keyword argument is `method`, which sets the IV method used to
estimate the model. Currently supported values are `2sls` (the default) and
`liml`.

.. code-block:: python

    # Imports and lodaing data
    y = 'wage'              # Dependent var
    X = ['educ']            # Endogenous regressor(s)
    Z = ['treatment']       # Instrumental variable(s)
    W = [ 'age', 'male']    # Exogenous regressor(s)

    results = mt.ivreg(df, y, X, Z, W)


Conley Errors (SHAC)
--------------------


Returned Results
----------------
