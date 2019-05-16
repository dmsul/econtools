from typing import Union, Generator, Optional, Any, Iterable

import string

import pandas as pd


def force_df(s: Union[pd.DataFrame, pd.Series],
             name: Optional[str]=None) -> pd.DataFrame:
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


def force_list(x: Any) -> list:
    """If type not `list`, pass to `force_interable`, then convert to list."""
    if isinstance(x, list):
        return x
    else:
        return list(force_iterable(x))


def force_iterable(x) -> Iterable:
    """If not iterable, wrap in tuple"""
    if hasattr(x, '__iter__') and type(x) is not str:
        return x
    else:
        return (x,)


def generate_chunks(
        iterable: Union[list, tuple],
        chunk_size: int) -> Generator[Union[list, tuple], None, None]:
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


def int2base(x: int, base: int) -> str:
    """
    Convert decimal x >= 0 to base <= 62

    Alpha values are case sensitive, with lowercase being higher value than
    upper case.
    """
    base_alphabet = _base62_alphabet()
    if base > len(base_alphabet):
        raise ValueError("Max base is 62. Passed base was {}".format(base))

    new_base = ''

    while x > 0:
        x, i = divmod(x, base)
        new_base = base_alphabet[i] + new_base

    return new_base


def base2int(x: str, base: int) -> int:
    """
    Convert x >= 0 of base `base` to decimal.

    Alpha values are case sensitive, with lowercase being higher value than
    upper case.
    """
    base62_alphabet = _base62_alphabet()
    if base > len(base62_alphabet):
        raise ValueError("Max base is 62. Passed base was {}".format(base))
    base_alphabet = base62_alphabet[:base]

    base10 = 0

    for place, value in enumerate(x[::-1]):
        values_base10 = base_alphabet.find(value)
        if values_base10 < 0:
            err_str = "Value `{}` is not a valid digit for base {}"
            raise ValueError(err_str.format(value, base))
        base10 += values_base10 * base ** place

    return base10


def _base62_alphabet() -> str:
    return string.digits + string.ascii_uppercase + string.ascii_lowercase
