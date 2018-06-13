.. currentmodule: econtools

.. _io:

*******************
Data (I/O) Tools
*******************

.. contents::
    :depth: 1
    :local:


``load_or_build``
-----------------

Basic Usage
~~~~~~~~~~~

In Python, a decorator is a special function that alters other functions. They
are invoked using ``@`` like so:


.. code-block:: python

    @decorator
    def my_function():
        print("I'm not sure I trust that instrument.")


:py:func:`~econtools.load_or_build` is a decorator that is used on functions that return
data (either a pandas Series or DataFrame) to cache that data to disk.

Let's say that you have function ``prep_data`` that preps a dataset for
regressions.

.. code-block:: python

    import numpy as np
    import pandas as pd

    def prep_data():
        df = pd.read_csv('my_data.csv')
        df['wage_sq'] = df['wage'] ** 2
        df['ln_wage'] = np.log(df['wage'])

        # Manipulate data in other ways...

        return df


Maybe this function takes a long time to run and you'd like to save it to your
hard drive so you don't have to re-process it every time. Instead of
copy/pasting the filepath associated with this function, you can just use
:py:func:`~econtools.load_or_build`:


.. code-block:: python

    import numpy as np
    import pandas as pd

    from econtools import load_or_build

    @load_or_build('reg_data.pkl')
    def prep_data():
        df = pd.read_csv('my_data.csv')
        df['wage_sq'] = df['wage'] ** 2
        df['ln_wage'] = np.log(df['wage'])

        # Manipulate data in other ways...

        return df


Here, ``@load_or_build('reg_data.pkl')`` adds the following
functionality to ``prep_data``:

#. If there's a file named ``'reg_data.pkl'``, load it and return it.
#. If that file doesn't exist, run ``prep_data`` like usual, save a copy to
   ``'reg_data.pkl'`` for next time then return the data.


Dynamic Filenames
~~~~~~~~~~~~~~~~~

Let's say your data function takes an argument, like ``year``:

.. code-block:: python

    def load_census_data(year):
        # Load data for year `year`...

        return df


In this case, you'll need a different file name on disk for each year.
:py:func:`~econtools.load_or_build` handles this using Python's named string
insertion using curly brackets like so:

.. code-block:: python

    @load_or_build('census_data_{year}.pkl')
    def load_census_data(year):
        # Load data for year `year`...

        return df

    if __name__ == '__main__':
        # Loads from 'census_data_2010.pkl'
        df = load_census_data(2010)


This works for both positional arguments and keywork arguments.


Special Keyword Switches
~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~econtools.load_or_build` adds two special keyword arguments to
functions it decorates.

* ``_rebuild`` (default ``False``): If ``_rebuild=True``,
  :py:func:`~econtools.load_or_build` will re-build the data output by the
  function and overwrite any saved version on disk.

* ``_load`` (default ``True``): If ``_load=False``,
  :py:func:`~econtools.load_or_build` will not look for saved data on disk and
  will only run the function as though you didn't use
  :py:func:`~econtools.load_or_build` in the first place.

Examples:


.. code-block:: python

    @load_or_build('census_data_{year}.pkl')
    def load_census_data(year):
        # Load data for year `year`...

        return df

    if __name__ == '__main__':
        # Loads from 'census_data_2010.pkl'
        df = load_census_data(2010)

        # Runs `load_census_data` and over writes what's on disk
        df = load_census_data(2010, _rebuild=True)

        # Doesn't load file on disk, only runs `load_census_data`
        df = load_census_data(2010, _load=True)


``save_cli``
------------

The function :py:func:`~econtools.save_cli` adds a ``--save`` flag to the
command line. When ``--save`` is included on the command line,
:py:func:`~econtools.save_cli` returns ``True``, and ``False`` otherwise.
This allows you to run a script without overwriting any tables or figures on
disk and avoid commenting/uncommenting lines of code that do the saving.


.. code-block:: python

    # script named "make_figure.py"

    from econtools import save_cli

    save = save_cli()

    if save:
        # Code to save the figure
    else:
        # Code to only display the figure


Then ``save`` switch is invoked on the command line using::

    $ python make_figure.py --save      # Saves figure
    $ python make_figure.py             # Does not save


``confirmer``
-------------

:py:func:`~econtools.confirmer` is a drop-in function to quickly allow a script
to get yes/no input from the user. It accepts a number of variations of
``yes``, ``Y``, ``YES``, etc., and will force a correct response by re-asking
the question if an invalid response is given.


.. code-block:: python

    # Script thermonuclear_war.py
    from econtools import confirmer

    question = "Shall we play a game?"

    answer = confirmer(question, default_no=True)

    if answer:
        # Action for 'yes' response
    else:
        # Action for no response


On the command line::

    $ python thermonuclear_war.py
    Shall we play a game? (y,[n]) >>> Y
    # Code executed for 'yes' response


``read`` and ``write``
----------------------

These function are primarily auxiliary functions used by
:py:func:`~econtools.load_or_build`, but they can be used directly if needed.

:py:func:`~econtools.read` will use the suffix of the passed filename to use
the correct ``pandas`` method to read the data.

.. code-block:: python

    from econtools import read

    df = read('my_data.csv')    # uses pandas.read_csv
    df = read('my_data.dta')    # uses pandas.read_stata
    df = read('my_data.pkl')    # uses pandas.read_pickle

:py:func:`~econtools.write` does the same, but with writing.
