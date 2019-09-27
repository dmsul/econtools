from typing import Optional, List

import pandas as pd
import numpy as np
import numpy.linalg as la

from econtools.util.gentools import force_list, force_df


def add_cons(df):
    df = df.copy()  # Protect original df

    if df.ndim == 1:
        df = pd.DataFrame(df)

    df['_cons'] = np.ones(df.shape[0])
    return df


def flag_sample(df, *args):
    varlist = []
    for var in args:
        if var is not None:
            varlist += force_list(var)
    sample = df[varlist].notnull().all(axis=1)
    return sample


def set_sample(df, sample, names):
    return tuple(_set_samp_core(df, sample, names))

def _set_samp_core(df, sample, names):
    for name in names:
        if name is None:
            yield None
        else:
            yield df.loc[sample, name].copy().reset_index(drop=True)


def demeaner(A, *args):
    return tuple(_demean_guts(A.squeeze(), args))

def _demean_guts(A, args):
    for df in args:
        # Ignore empty `df` (e.g. empty list of exogenous included regressors)
        if df is None or df.empty:
            yield df
        else:
            group_name = A.name
            mean = df.groupby(A).mean()
            large_mean = force_df(A).join(mean, on=group_name).drop(group_name,
                                                                    axis=1)
            if df.ndim == 1:
                large_mean = large_mean.squeeze()
            demeaned = df - large_mean
            yield demeaned


def unpack_shac_args(argdict):
    if argdict is None:
        return None, None, None, None

    # Check if extra args passed (Do NOT alter `argdict` with `pop`!)
    extra_args = set(argdict.keys()).difference(set(('x', 'y', 'band', 'kern')))
    if extra_args:
        err_str = 'Extra `shac` args: {}'
        raise ValueError(err_str.format(tuple(extra_args)))

    shac_x = argdict['x']
    shac_y = argdict['y']
    shac_band = argdict['band']
    shac_kern = argdict['kern']

    return shac_x, shac_y, shac_band, shac_kern


def flag_nonsingletons(df, avar, sample):
    """Boolean flag for 'not from a singleton `avar` group."""
    counts = df[sample].groupby(avar).size()
    big_counts = df[[avar]].join(counts.to_frame('_T'), on=avar).fillna(0)
    non_single = big_counts['_T'] > 1
    return non_single


def find_colinear_columns(
        arr: np.ndarray,
        arr_rank: Optional[int]=None) -> List[int]:

    if arr_rank is None:
        arr_rank = la.matrix_rank(arr)
    K = arr.shape[1]
    num_colinear_cols = K - arr_rank
    if num_colinear_cols <= 0: raise ValueError("No colinear columns.")

    # Cycle through all columns; if a col doesn't increase rank, it's
    # colinear so flag it
    target_rank = 2
    colinear_cols = []
    for j in range(1, K):
        sub_rank = la.matrix_rank(arr[:, :(j + 1)])
        if sub_rank == target_rank:
            target_rank += 1
        else:
            colinear_cols.append(j)

        if len(colinear_cols) == num_colinear_cols:
            break  # We found them all; quit.

    return colinear_cols
