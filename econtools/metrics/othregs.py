import pandas as pd
import numpy as np
import scipy.linalg as la

from regutil import force_list, demeaner, flag_nonsingletons
from core import (fitguts, inference, add_cons, flag_sample, unpack_spatialargs,
                  Results)


def areg(df, y, x, avar=None, vce_type=None, cluster=None, nosingles=True,
         spatial_hac=None):

    if avar is None:
        raise ValueError("Must pass string `avar`, not {}".format(type(avar)))

    # Unpack spatial HAC args
    sp_args = unpack_spatialargs(spatial_hac)
    spatial_x, spatial_y, spatial_band, spatial_kern = sp_args
    if spatial_x is not None:
        vce_type = 'spatial'

    xname = force_list(x)
    basic_sample = flag_sample(df, y, xname, cluster, avar, spatial_x,
                               spatial_y)
    if nosingles:
        not_singleton = flag_nonsingletons(df, avar)
        sample = (basic_sample) & (not_singleton)
    else:
        sample = basic_sample

    Y = df.loc[sample, y].copy().reset_index(drop=True)
    X = df.loc[sample, xname].copy().reset_index(drop=True)
    A = df.loc[sample, avar].copy().reset_index(drop=True)
    if cluster is not None:
        cluster_id = df.loc[sample, cluster].copy()
    else:
        cluster_id = None

    if spatial_x is not None:
        space_x = df.loc[sample, spatial_x].copy()
        space_y = df.loc[sample, spatial_y].copy()
    else:
        space_x, space_y = None, None

    # Demean
    y_demean = demeaner(Y, A)
    x_demean, x_means = demeaner(X, A, return_mean=True)

    # Get main results
    results = fitguts(y_demean, x_demean)
    # Correct properties made weird by demeaning
    results.sst = Y
    results._nocons = True

    # Do inference
    N, K = X.shape
    K += len(A.unique())    # Adjust dof's for group means

    inferred = inference(y_demean, x_demean, results.xpxinv, results.beta, N, K,
                         vce_type=vce_type, cluster=cluster_id,
                         spatial_x=space_x, spatial_y=space_y,
                         spatial_band=spatial_band,
                         spatial_kern=spatial_kern)
    results.__dict__.update(**inferred)

    return results


def tsls(df, y, x_end, z, x_exog, vce_type=None, cluster=None, addcons=False,
         spatial_hac=None):
    """
    `z` is a list of names in `x` that are the excluded instruments.
    `x_end` is a list of names in `x` that are endogenous X's.
    """

    # Unpack spatial HAC args
    sp_args = unpack_spatialargs(spatial_hac)
    spatial_x, spatial_y, spatial_band, spatial_kern = sp_args
    if spatial_x is not None:
        vce_type = 'spatial'

    x_end_name = force_list(x_end)
    z_name = force_list(z)
    x_exog_name = force_list(x_exog)
    sample = flag_sample(df, y, x_end_name, z_name, x_exog_name, cluster)

    Y = df.loc[sample, y].copy().reset_index(drop=True)
    X_end = df.loc[sample, x_end_name].copy().reset_index(drop=True)
    X_exog = df.loc[sample, x_exog_name].copy().reset_index(drop=True)
    Z = df.loc[sample, z_name].copy().reset_index(drop=True)
    if cluster is not None:
        cluster_id = df.loc[sample, cluster].copy()
    else:
        cluster_id = None

    if spatial_x is not None:
        space_x = df.loc[sample, spatial_x].copy()
        space_y = df.loc[sample, spatial_y].copy()
    else:
        space_x, space_y = None, None

    if addcons:
        X_exog = add_cons(X_exog)

    # First Stage
    X_hat = _first_stage(X_end, X_exog, Z)

    results = fitguts(Y, X_hat)

    # R^2 doesn't mean anything in IV/2SLS
    results._r2 = np.nan
    results._r2_a = np.nan

    # Do inference
    N, K = X_hat.shape
    inferred = inference(Y, X_hat, results.xpxinv, results.beta, N, K,
                         x_for_resid=pd.concat((X_end, X_exog), axis=1),
                         vce_type=vce_type, cluster=cluster_id,
                         spatial_x=space_x, spatial_y=space_y,
                         spatial_band=spatial_band,
                         spatial_kern=spatial_kern)
    results.__dict__.update(**inferred)

    return results


def _first_stage(x_end, x_exog, Z):
    X_hat = pd.concat((x_end, x_exog), axis=1)
    all_exog = pd.concat((Z, x_exog), axis=1)
    for an_x in x_end.columns:
        this_x = x_end[an_x]
        first_stage = fitguts(this_x, all_exog)
        X_hat[an_x] = np.dot(all_exog, first_stage.beta)
    return X_hat


def liml(df, y_name, x_name, z_name, w_name, a_name=None,
         vce_type=None, cluster=None, spatial_hac=None,
         addcons=False, _kappa_debug=None,
         nosingles=True):
    """
    `z` is a list of names in `x` that are the excluded instruments.
    `x_end` is a list of names in `x` that are endogenous X's.
    """

    if addcons and a_name:
        raise ValueError("Can't add a constent when demeaning!")

    # Unpack spatial HAC args
    sp_args = unpack_spatialargs(spatial_hac)
    spatial_x, spatial_y, spatial_band, spatial_kern = sp_args
    if spatial_x is not None:
        vce_type = 'spatial'

    x_name = force_list(x_name)
    z_name = force_list(z_name)
    w_name = force_list(w_name)
    sample = flag_sample(df, y_name, x_name, z_name, w_name, a_name,
                         cluster, spatial_x, spatial_y)
    if nosingles and a_name:
        not_singleton = flag_nonsingletons(df, a_name)
        sample &= not_singleton

    y = df.loc[sample, y_name].copy().reset_index(drop=True)
    x = df.loc[sample, x_name].copy().reset_index(drop=True)
    w = df.loc[sample, w_name].copy().reset_index(drop=True)
    z = df.loc[sample, z_name].copy().reset_index(drop=True)
    if a_name:
        A = df.loc[sample, a_name].copy().reset_index(drop=True)
        y_raw = y.copy()
        y = demeaner(y, A)
        x = demeaner(x, A)
        w = demeaner(w, A)
        z = demeaner(z, A)
    else:
        A = None
        y_raw = y

    if cluster is not None:
        cluster_id = df.loc[sample, cluster].copy()
        vce_type = 'cluster'
    else:
        cluster_id = None

    if spatial_x is not None:
        space_x = df.loc[sample, spatial_x].copy()
        space_y = df.loc[sample, spatial_y].copy()
    else:
        space_x, space_y = None, None

    if addcons:
        w = add_cons(w)

    # LIML prep
    Z = pd.concat((z, w), axis=1)
    kappa, ZZ_inv = _liml(y, x, w, Z)
    X = pd.concat((x, w), axis=1)
    # Solve system
    XX = X.T.dot(X)
    XZ = X.T.dot(Z)
    Xy = X.T.dot(y)
    Zy = Z.T.dot(y)

    # When `kappa` = 1 is 2sls, `kappa` = 0 is OLS
    if _kappa_debug is not None:
        kappa = _kappa_debug

    xpxinv = la.inv(
        (1-kappa)*XX + kappa*np.dot(XZ.dot(ZZ_inv), XZ.T)
    )
    xpy = (1-kappa)*Xy + kappa*np.dot(XZ.dot(ZZ_inv), Zy)
    beta = pd.Series(xpxinv.dot(xpy).squeeze(), index=X.columns)

    results = Results(beta=beta, xpxinv=xpxinv)
    results.sst = y_raw
    results.kappa = kappa

    # R^2 doesn't mean anything in IV/2SLS
    results._r2 = np.nan
    results._r2_a = np.nan

    # Do inference
    N, K = X.shape
    if a_name:
        K += A.unique().shape[0]

    if vce_type is None:
        se_xpxinv = xpxinv
    else:
        se_xpxinv = xpxinv.dot(XZ).dot(ZZ_inv)

    inferred = inference(y, Z, se_xpxinv, results.beta, N, K,
                         x_for_resid=X,
                         vce_type=vce_type, cluster=cluster_id,
                         spatial_x=space_x, spatial_y=space_y,
                         spatial_band=spatial_band,
                         spatial_kern=spatial_kern, cols=X.columns)
    results.__dict__.update(**inferred)

    return results


def _liml(y, x, w, Z):
    Y = pd.concat((y, x), axis=1).astype(np.float64)
    YY = Y.T.dot(Y)
    YZ = Y.T.dot(Z)
    ZZ_inv = la.inv(Z.T.dot(Z))

    bread = la.inv(la.sqrtm(
        YY - np.dot(YZ.dot(ZZ_inv), YZ.T)
    ))

    if not w.empty:
        Yw = Y.T.dot(w)
        ww_inv = la.inv(w.T.dot(w))
        meat = YY - np.dot(Yw.dot(ww_inv), Yw.T)
    else:
        meat = YY

    eigs = la.eigvalsh(bread.dot(meat).dot(bread))
    kappa = np.min(eigs)
    return kappa, ZZ_inv


def atsls(df, y, x_end, z, x_exog, vce_type=None, cluster=None, avar=None,
          nosingles=True, spatial_hac=None):

    if type(avar) is not str:
        raise ValueError("Must pass string `avar`, not {}".format(type(avar)))

    # Unpack spatial HAC args
    sp_args = unpack_spatialargs(spatial_hac)
    spatial_x, spatial_y, spatial_band, spatial_kern = sp_args

    x_end_name = force_list(x_end)
    z_name = force_list(z)
    x_exog_name = force_list(x_exog)

    basic_sample = flag_sample(df, y, x_end_name, z_name, x_exog_name, cluster,
                               avar)
    if nosingles:
        not_singleton = flag_nonsingletons(df, avar)
        sample = (basic_sample) & (not_singleton)
    else:
        sample = basic_sample

    # Separate variables, restrict to sample
    Y = df.loc[sample, y].copy().reset_index(drop=True)
    X_end = df.loc[sample, x_end_name].copy().reset_index(drop=True)
    X_exog = df.loc[sample, x_exog_name].copy().reset_index(drop=True)
    A = df.loc[sample, avar].copy().reset_index(drop=True)
    Z = df.loc[sample, z_name].copy().reset_index(drop=True)
    if cluster is not None:
        cluster_id = df.loc[sample, cluster].copy()
    else:
        cluster_id = None

    if spatial_x is not None:
        space_x = df.loc[sample, spatial_x].copy()
        space_y = df.loc[sample, spatial_y].copy()
    else:
        space_x, space_y = None, None

    # Demean
    y_demean = demeaner(Y, A)
    x_end_demean = demeaner(X_end, A)
    x_exog_demean = demeaner(X_exog, A)
    z_demean = demeaner(Z, A)

    # First Stage
    X_hat = _first_stage(x_end_demean, x_exog_demean, z_demean)

    results = fitguts(y_demean, X_hat)
    results.sst = Y
    results._nocons = True
    results._r2 = np.nan
    results._r2_a = np.nan

    # Do inference
    N, K = X_hat.shape
    K += len(A.squeeze().unique())

    inferred = inference(y_demean, X_hat, results.xpxinv, results.beta, N, K,
                         x_for_resid=pd.concat((x_end_demean, x_exog_demean),
                                               axis=1),
                         vce_type=vce_type, cluster=cluster_id,
                         spatial_x=space_x, spatial_y=space_y,
                         spatial_band=spatial_band,
                         spatial_kern=spatial_kern)

    results.__dict__.update(**inferred)

    return results


if __name__ == '__main__':
    from os import path
    test_path = path.split(path.relpath(__file__))[0]
    data_path = path.join(test_path, 'tests', 'data')
    if 1 == 1:
        df = pd.read_stata(path.join(data_path, 'auto.dta'))
        for col in df.columns:
            try:
                df[col] = df[col].astype(np.float64)
            except:
                pass
        df['foreign'] = df['foreign'].cat.codes
        y = 'price'
        x_end = ['mpg', 'length']
        x_end = ['mpg']
        z = ['weight', 'trunk', 'gear_ratio']
        x_exog = 'length'
        # x_exog = []
        rhv = x_end + z
        cluster = 'gear_ratio'
        if 0 == 1:
            results = atsls(df, y, x_end, z, x_exog, avar=cluster,
                            nosingles=False,
                            # vce_type='robust',
                            )
        elif 1 == 1:
            results = liml(df, y, x_end, z, x_exog,
                           # addcons=True,
                           a_name='foreign',
                           # vce_type='robust',
                           # cluster=cluster,
                           # _kappa_debug=1.0000516,
                           )
            results2 = tsls(df, y, x_end, z, x_exog, addcons=True,)
            # results2 = reg(df, y, x_end, addcons=True,)
        else:
            results = areg(df, y, x_end, avar=cluster,
                           # cluster=cluster
                           )
        stata_a_foreign = np.array([
            [14751.4337665404, 27364.1521779233],
            [27364.1521779233,  135444.4923135681]])

        stata_a_gear = np.array([
            [16382.8783882600, 26148.3546596647],
            [26148.3546596647,  91844.9340613835]])
        try:
            ratios_a_foreign = results.vce / stata_a_foreign
            ratios_a_gear = results.vce / stata_a_gear
            if 1 == 1:
                print ratios_a_gear
            else:
                print ratios_a_foreign
        except:
            pass
    else:
        df = pd.read_stata(path.join(data_path, 'nlswork.dta'))
        y_name = 'ln_wage'
        rhv = ['age', 'tenure', 'collgrad']
        cluster_name = 'race'
        df = df[[y_name] + [cluster_name] + rhv].dropna()
        df['race'] = df['race'].cat.codes
        y = df['ln_wage']
        cluster = df['race']
        results = atsls(y, df[rhv], x_end='tenure', z='age', avar=cluster,
                        vce_type='hc1', cluster=None)
    print results.summary
    print results2.summary
