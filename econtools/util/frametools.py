from typing import Optional, Union, Iterable, cast

import pandas as pd
import numpy as np

from econtools.util.gentools import force_iterable


def stata_merge(left: pd.DataFrame, right: pd.DataFrame,
                assertval: Optional[int] = None, gen: str = '_m',
                **kwargs) -> pd.DataFrame:
    """Merge two DataFrames via ``pandas.merge`` but with some additional
    features. Specifically, an additional column is added to the returned
    DataFrame with the default label ``'_m'``. For each row of the returned
    DataFrame, ``'_m'`` equals 1 if the row existed only in ``left``, 2 if the
    row exited only in ``right``, and 3 if it exists in both, i.e., was
    successfully merged.

    Args:
        left (DataFrame): Left DataFrame to merge.
        right (DataFrame): Right DataFrame to merge.

    Keyword Args:
        assertval (int): Assert that all values of ``'_m'`` are equal to
            ``assertval``. Under default (``None``) and no assertion is made.
        gen (str): Name of the merge status variable. Default is ``'_m'``.
        kwargs: Any standard keyword arg for ``pandas.merge``, such as ``on``
            or ``how``.

    Returns:
        :py:class:`pandas.DataFrame`: A ``DataFrame`` that is the merged output
        of ``left`` and ``right``.
    """
    # Tmp variables needed for merge status variable
    left_tmp = 'tmpa'
    right_tmp = 'tmpb'
    while left_tmp in left.columns:
        left_tmp += 'a'
    while right_tmp in right.columns:
        right_tmp += 'b'
    # Try this copying stuff to avoid altering the original dataframes
    copy_left, copy_right = left.copy(), right.copy()
    copy_left[left_tmp] = 1
    copy_right[right_tmp] = 2
    # Actual merge
    new = pd.merge(copy_left, copy_right, **kwargs)
    # Generate merge status flag
    new[gen] = new[left_tmp].add(new[right_tmp], fill_value=0)
    # Clean up tmp variables
    del new[left_tmp], new[right_tmp]

    # Show distribution of rows by merge status
    if assertval:
        try:
            assert (new[gen] == assertval).min()
        except AssertionError:
            print("Merge assertion is false!")
            print(new.groupby(gen).size() / new.shape[0])
            raise
        else:
            del new[gen]
    else:
        print(new.groupby(gen).size() / new.shape[0])

    return new


def group_id(df: pd.DataFrame,
             cols: Optional[list] = None,
             name: str = 'group_id',
             merge: bool = False) -> pd.DataFrame:
    """Generate a unique integer ID from a DataFrame or columns of the
    DataFrame. Specifically, create a unique number for every combination

    Args:
        df (DataFrame): DataFrame of interest.

    Keyword Args:
        cols (list): List of columns to use for ID generation. Default
          (``None``) uses all columns in ``df``.
        name (str): Name of the new ID variable. Default is ``'group_id'``.
        merge (bool): Return the full input DataFrame `df` with the new group
          ID column merged on. Default is ``False``.

    Returns:
        :py:class:`pandas.DataFrame`:A ``DataFrame`` with the new group ID and
        ``cols`` if ``merge=False``, or if ``merge=True``, the input
        ``DataFrame`` with group ID merged on as a new column.
    """
    if not cols:
        cols = df.columns.tolist()

    if name in df.columns:
        raise ValueError("ID name '{}' is a column name.".format(name))

    unique_df = df[cols].drop_duplicates().reset_index(drop=True)
    unique_df.index.name = name
    unique_df = unique_df.reset_index()

    if merge:
        unique_df = stata_merge(df, unique_df, on=cols, how='left',
                                assertval=3)
        new_i, new_j = unique_df.shape
        old_i, old_j = df.shape
        assert new_i == old_i and new_j == old_j + 1
        unique_df.index = df.index

    return unique_df


def winsorize(df: pd.DataFrame, by: Union[str, Iterable[str]],
              p: Iterable[Union[tuple, float]]=(.01, .99)):
    """Drop variables in `by' outside quantiles `p`."""
    # TODO: Some kind of warning/error if too fine of quantiles are
    #       requested for the number of rows, e.g. .99 with 5 rows.
    df = df.copy()

    by = cast(Iterable[str], force_iterable(by))

    # Allow different cutoffs for different variables
    if hasattr(p[0], '__iter__'):
        assert len(p) == len(by)
    else:
        p = [p] * len(by)

    survive_winsor = np.array([True] * df.shape[0])

    for idx, col in enumerate(by):
        cuts = df[col].quantile(p[idx]).values
        survive_this = np.logical_and(df[col] >= cuts[0], df[col] <= cuts[1])
        survive_winsor = np.minimum(survive_winsor, survive_this)

    df = df[survive_winsor]

    return df


def df_to_list(df: Union[list, pd.DataFrame]) -> list:
    """ Turn rows of DataFrame to list of Series objects """
    if isinstance(df, list):
        return df
    elif isinstance(df, pd.DataFrame):
        return [b for a, b in df.iterrows()]
    else:
        raise ValueError
