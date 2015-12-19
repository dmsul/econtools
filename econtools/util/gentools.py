import pandas as pd


def force_df(s, name=None):
    """
    Forces a Series to a DataFrame. DataFrames are returned unaffected. Other
    objects raise `ValueError`.
    """
    # Check if DF or Series
    if isinstance(s, pd.core.frame.DataFrame):
        return s
    elif not isinstance(s, pd.core.series.Series):
        raise ValueError("`s` is a {}".format(type(s)))
    else:
        return s.to_frame(name)


def force_list(x):
    """If type not `list`, pass to `force_interable`, then convert to list."""
    if isinstance(x, list):
        return x
    else:
        return list(force_iterable(x))


def force_iterable(x):
    """If not iterable, wrap in tuple"""
    if hasattr(x, '__iter__'):
        return x
    else:
        return (x,)


def generate_chunks(iterable, chunk_size):
    """Go through `iterable` one chunk at a time."""
    length = len(iterable)
    N_chunks = length // chunk_size
    runs = 0
    while runs <= N_chunks:
        i, j = chunk_size*runs, chunk_size*(runs + 1)
        if runs < N_chunks:
            yield iterable[i:j]
        elif i < length:
            yield iterable[i:]
        else:
            pass  # Don't return an empty last list
        runs += 1
