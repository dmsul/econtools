import pandas as pd
import numpy as np

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
