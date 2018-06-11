.. _overview:

.. currentmodule: econtools

****************
Package overview
****************

:mod:`econtools` consists of several sets of tools commonly used for
econometrics and data manipulation.

#. :ref:`overview.econometrics`
#. :ref:`overview.io`

.. _overview.econometrics:

Econometric Tools
-----------------

.. Overview
.. Import Usage
.. OLS tutorial
.. Link to/include API

The econometrics tools in :mod:`econtools` include

* Common regression techniques (OLS, 2SLS, LIML)
* Option to absorb any variable via within-transformation. This is similar to
  the ``areg, absorb`` command in Stata but it can be used with any relevant
  regression command. This consolidates ``reg``, ``areg``, ``xtreg``, etc.
* Robust standard errors

  * HAC (robust/HC1, HC2, HC3)
  * Clustered standard errors
  * Spatial HAC (aka SHAC or Conley standard errors) with uniform and triangle
    kernels

* F-tests by variable name or arbitrary `R` matrix.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import econtools.metrics as mt

    # Load a data file with columns 'ln_wage', 'educ', and 'state'
    df = pd.read_csv('my_data.csv')

    # Regress log wage on education
    results_ols = mt.reg(df, 'ln_wage', 'educ')

    # Use robust standard errors
    results_robust = mt.reg(df, 'ln_wage', 'educ', vce_type='robust')

    # Cluster standard errors by state
    results_cluster = mt.reg(df, 'ln_wage', 'educ', cluster='state')

    # Add state fixed effects via within transform (demeaning)
    results_areg = mt.reg(df, 'ln_wage', 'educ', a_name='state')

    # Display results
    print results_areg.summary


And that's it.


Advanced Usage
~~~~~~~~~~~~~~

See :ref:`metricsadv`.

.. _overview.io:

Input/Output
------------

:mod:`econtools` contains a number of boilerplate methods that facilitate
creating datasets and saving them to disk in flexible ways.
The most signifiacnt of these is ``load_or_build``, a decorator that can be
used to cache datasets to disk.

For example, let's say that you have function ``prep_for_regs`` that preps some
other dataset for regressions. If this function takes a long time to run, you may
want to save this dataset so that I don't have re-create the data every single
time you run a new regression.

.. code-block:: python

    import pandas as pd
    from econtools import load_or_build

    @load_or_build('reg_ready_data.pkl')
    def prep_for_regs():
        """ Prep data for regressions. """

        df = pd.read_csv('my_data.csv')
        # Create 'ln_wage' from 'wage'
        df['wage_sq'] = df['wage'] ** 2
        # Manipulate data in other ways...

        return df
