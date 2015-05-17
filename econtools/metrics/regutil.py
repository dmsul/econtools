import pandas as pd
import numpy as np
from patsy.contrasts import Treatment


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

def _set_samp_core(df, sample, names):      #noqa
    for name in names:
        if name is None:
            yield None
        else:
            yield df.loc[sample, name].copy().reset_index(drop=True)


def demeaner(A, *args):
    return tuple(_demean_guts(A.squeeze(), args))

def _demean_guts(A, args):      #noqa
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


def unpack_spatialargs(argdict):
    if argdict is None:
        return None, None, None, None
    spatial_x = argdict.pop('x', None)
    spatial_y = argdict.pop('y', None)
    spatial_band = argdict.pop('band', None)
    spatial_kern = argdict.pop('kern', None)
    if argdict:
        raise ValueError("Extra args passed: {}".format(argdict.keys()))
    return spatial_x, spatial_y, spatial_band, spatial_kern


def flag_nonsingletons(df, avar):
    """Boolean flag for 'not from a singleton `avar` group."""
    counts = df.groupby(avar).size()
    big_counts = df[[avar]].join(counts.to_frame('_T'), on=avar)
    non_single = big_counts['_T'] > 1
    return non_single


def force_df(df):
    if df.ndim == 1:
        df = pd.DataFrame(df.copy())
    return df


def force_list(x):
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


def windsorize(df, by, p=(.01, .99)):
    """Drop variables in `by' outside quantiles `p`."""
    # TODO: Move to utils? Mysci?
    # TODO: Some kind of warning/error if too fine of quantiles are
    #       requested for the number of rows, e.g. .99 with 5 rows.
    df = df.copy()

    by = force_iterable(by)

    # Allow different cutoffs for different variables
    if hasattr(p[0], '__iter__'):
        assert len(p) == len(by)
    else:
        p = [p] * len(by)

    survive_windsor = np.array([True] * df.shape[0])

    for idx, col in enumerate(by):
        cuts = df[col].quantile(p[idx]).values
        survive_this = np.logical_and(df[col] >= cuts[0], df[col] <= cuts[1])
        survive_windsor = np.minimum(survive_windsor, survive_this)

    df = df[survive_windsor]

    return df


class PatsyForceOmit(object):

    """Force patsy to omit a category even if it doesn't want to."""

    def __init__(self):
        pass

    def code_with_intercept(self, levels):
        return Treatment(reference=0).code_without_intercept(levels)

    def code_without_intercept(self, levels):
        return Treatment(reference=0).code_without_intercept(levels)
