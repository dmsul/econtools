from __future__ import division

import pandas as pd
import numpy as np
import numpy.linalg as la    # scipy.linalg yields slightly diff results (tsls)
from numpy.linalg import matrix_rank        # not in `scipy.linalg`
from scipy.linalg import sqrtm              # notin `numpy.linalg`

import scipy.stats as stats

from econtools.util import force_list
from regutil import (add_cons, flag_sample, set_sample, flag_nonsingletons,
                     demeaner, unpack_spatialargs)


def ivreg(df, y_name, x_name, z_name, w_name,
          a_name=None, nosingles=True,
          method='2sls', _kappa_debug=None,
          vce_type=None, cluster=None, spatial_hac=None,
          addcons=None, nocons=False):
    # Set `vce_type`
    vce_type = _set_vce_type(vce_type, cluster, spatial_hac)
    # Unpack spatial HAC args
    sp_args = unpack_spatialargs(spatial_hac)
    spatial_x, spatial_y, spatial_band, spatial_kern = sp_args
    # Handle names
    x_name = force_list(x_name)
    w_name = force_list(w_name)
    # Flag and restrict sample
    sample_cols = (y_name, x_name, a_name, cluster, spatial_x, spatial_y,
                   w_name, z_name)
    sample = flag_sample(df, *sample_cols)
    if nosingles and a_name:
        sample &= flag_nonsingletons(df, a_name)
    y, x, A, cluster_id, space_x, space_y, w, z = set_sample(df, sample,
                                                             sample_cols)

    if addcons and not a_name:
        w = add_cons(w)

    # Demean and save true `y` if needed
    if a_name:
        y_raw = y.copy()
        y, x, z, w = demeaner(A, y, x, z, w)
    else:
        y_raw = y

    # Estimation
    if method == '2sls':
        sum_in_sandwich = _first_stage(x, w, z)
        results = fitguts(y, sum_in_sandwich)
    elif method == 'liml':
        results, sum_in_sandwich, kappa = _liml(y, x, z, w, _kappa_debug,
                                                vce_type)
        results.kappa = kappa
    else:
        raise ValueError("IV method '{}' not supported".format(method))

    results._r2 = np.nan
    results._r2_a = np.nan

    N = y.shape[0]
    K = x.shape[1]
    try:
        K += w.shape[1]
    except:
        pass

    # Corrections
    if a_name:
        if _fe_not_nested_cluster(cluster_id, A):
            K += len(A.unique())    # Adjust dof's for group means
        results.sst = y_raw
        results._nocons = True
    else:
        results.__dict__.update(dict(_nocons=nocons))

    inferred = inference(y, sum_in_sandwich, results.xpxinv, results.beta, N, K,
                         x_for_resid=pd.concat((x, w), axis=1),
                         vce_type=vce_type, cluster=cluster_id,
                         spatial_x=space_x, spatial_y=space_y,
                         spatial_band=spatial_band,
                         spatial_kern=spatial_kern,
                         cols=x.columns.tolist() + w.columns.tolist())
    results.__dict__.update(**inferred)

    results.sample = sample

    return results


def _first_stage(x, w, z):
    X_hat = pd.concat((x, w), axis=1)
    Z = pd.concat((z, w), axis=1)
    for an_x in x.columns:
        this_x = x[an_x]
        first_stage = fitguts(this_x, Z)
        X_hat[an_x] = np.dot(Z, first_stage.beta)
    return X_hat


def _liml(y, x, z, w, _kappa_debug, vce_type):
    Z = pd.concat((z, w), axis=1)
    kappa, ZZ_inv = _liml_kappa(y, x, w, Z)
    X = pd.concat((x, w), axis=1)
    # Solve system
    XX = X.T.dot(X)
    XZ = X.T.dot(Z)
    Xy = X.T.dot(y)
    Zy = Z.T.dot(y)

    # When `kappa` = 1 is 2sls, `kappa` = 0 is OLS
    if _kappa_debug is not None:
        kappa = _kappa_debug
    # If exactly identified, same as 2sls, make it so
    elif x.shape[1] == z.shape[1]:
        kappa = 1

    xpxinv = la.inv(
        (1-kappa)*XX + kappa*np.dot(XZ.dot(ZZ_inv), XZ.T)
    )
    xpy = (1-kappa)*Xy + kappa*np.dot(XZ.dot(ZZ_inv), Zy)
    beta = pd.Series(xpxinv.dot(xpy).squeeze(), index=X.columns)

    # LIML uses non-standard 'bread' in the sandwich estimator
    if vce_type is None:
        se_xpxinv = xpxinv
    else:
        se_xpxinv = xpxinv.dot(XZ).dot(ZZ_inv)

    results = Results(beta=beta, xpxinv=se_xpxinv)

    return results, Z, kappa

def _liml_kappa(y, x, w, Z):        #noqa
    Y = pd.concat((y, x), axis=1).astype(np.float64)
    YY = Y.T.dot(Y)
    YZ = Y.T.dot(Z)
    ZZ_inv = la.inv(Z.T.dot(Z))

    bread = la.inv(sqrtm(
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


def reg(df, y_name, x_name,
        a_name=None, nosingles=True,
        vce_type=None, cluster=None, spatial_hac=None,
        addcons=None, nocons=False,
        ):
    """
    Parameters
    ----------
    df  - DataFrame, contains all data that will be used in the regression.
    y   - string, name of outcome variable
    x   - list, name(s) of regressors

    vce_type - string, type of variance-covariance estimator. Options are
        ['robust', 'hc1', 'hc2', 'hc3', 'cluster']. 'robust' and 'hc1' do the
        same thing; 'cluster' requires kwargs `cluster` and is not necessary.
    cluster - Series, cluster ID's
    nocons  - boolean, the regression does not have a constant
    addcons - boolean, add a column of 1's to the list of regressors

    Output
    ------
    Regression output object
    """
    # Set `vce_type`
    vce_type = _set_vce_type(vce_type, cluster, spatial_hac)
    # Unpack spatial HAC args
    sp_args = unpack_spatialargs(spatial_hac)
    spatial_x, spatial_y, spatial_band, spatial_kern = sp_args
    # Handle names
    x_name = force_list(x_name)
    # Flag and restrict sample
    sample_cols = (y_name, x_name, a_name, cluster, spatial_x, spatial_y)
    sample = flag_sample(df, *sample_cols)
    if nosingles and a_name:
        sample &= flag_nonsingletons(df, a_name)
    y, X, A, cluster_id, space_x, space_y = set_sample(df, sample, sample_cols)

    if addcons:
        X = add_cons(X)

    if a_name:
        y_raw = y.copy()
        y, X = demeaner(A, y, X)
    else:
        y_raw = y

    results = fitguts(y, X)

    N, K = X.shape

    # Corrections
    if a_name:
        if _fe_not_nested_cluster(cluster_id, A):
            K += len(A.unique())    # Adjust dof's for group means
        results.sst = y_raw
        results._nocons = True
    else:
        results.__dict__.update(dict(_nocons=nocons))

    inferred = inference(y, X, results.xpxinv, results.beta, N, K,
                         vce_type=vce_type, cluster=cluster_id,
                         spatial_x=space_x, spatial_y=space_y,
                         spatial_band=spatial_band,
                         spatial_kern=spatial_kern)
    results.__dict__.update(**inferred)

    results.sample = sample

    return results


def _set_vce_type(vce_type, cluster, spatial_hac):
    """ Check for argument conflicts, then set `vce_type` if needed.  """
    # Check for valid arg
    valid_vce = (None, 'robust', 'hc1', 'hc2', 'hc3', 'cluster', 'spatial')
    if vce_type not in valid_vce:
        raise ValueError("VCE type '{}' is not supported".format(vce_type))
    # Check for conflicts
    cluster_err = cluster and (vce_type != 'cluster' and vce_type is not None)
    shac_err = spatial_hac and (vce_type != 'spatial' and vce_type is not None)
    if (cluster and spatial_hac) or cluster_err or shac_err:
        raise ValueError("VCE type conflict!")
    # Set `vce_type`
    if cluster:
        new_vce = 'cluster'
    elif spatial_hac:
        new_vce = 'spatial'
    else:
        new_vce = vce_type

    return new_vce


def _fe_not_nested_cluster(cluster_id, A):
    """ Check if FE's are nested within clusters (affects DOF correction)."""
    if (cluster_id is None) or (A is None) or (cluster_id.name == A.name):
        return True
    else:
        joint = pd.concat((cluster_id, A), axis=1)
        names = [cluster_id.name, A.name]
        pair_counts = joint.groupby(names)[A.name].count()
        num_of_clusters = pair_counts.groupby(level=A.name).count()
        return num_of_clusters.max() != 1


def fitguts(y, x):
    """Checks dimensions, inverts, instantiates `Results'"""
    # Y should be 1D
    assert y.ndim == 1
    # X should be 2D
    assert x.ndim == 2

    xpxinv = la.inv(np.dot(x.T, x))
    xpy = np.dot(x.T, y)
    beta = pd.Series(np.dot(xpxinv, xpy).squeeze(), index=x.columns)

    results = Results(beta=beta, xpxinv=xpxinv)
    results.sst = y     # Passes `y` to sst setter, doesn't save `y`

    return results


def inference(y, x, xpxinv, beta, N, K, vce_type, cluster, x_for_resid=None,
              spatial_x=None, spatial_y=None, spatial_band=None,
              spatial_kern=None, cols=None):

    if x_for_resid is not None:
        yhat = np.dot(x_for_resid, beta)
    else:
        yhat = np.dot(x, beta)
    resid = y - yhat

    if cluster is not None:
        g = len(pd.value_counts(cluster))
        t_df = g - 1
        vce_type = 'cluster'
    elif spatial_x is not None:
        if vce_type is None:
            vce_type = 'spatial'
        g = None
        t_df = N - K    # XXX: No idea what this should be
    else:
        t_df = N - K
        g = None

    vce = robust_vce(vce_type, xpxinv, x, resid, N, K, cluster=cluster, g=g,
                     spatial_x=spatial_x, spatial_y=spatial_y,
                     spatial_band=spatial_band,
                     spatial_kern=spatial_kern,
                     cols=cols)
    se = pd.Series(np.sqrt(np.diagonal(vce)), index=vce.columns)

    t_stat = beta.div(se)
    # Set t df
    pt = pd.Series(
        stats.t.cdf(-np.abs(t_stat), t_df)*2,  # `t.cdf` is P(x<X)
        index=vce.columns
    )
    conf_level = .95
    crit_value = stats.t.ppf(conf_level + (1 - conf_level)/2, t_df)
    ci_lo = beta - crit_value*se
    ci_hi = beta + crit_value*se
    ret_dict = dict(resid=resid, vce=vce, vce_type=vce_type, se=se,
                    t_stat=t_stat, pt=pt, ci_lo=ci_lo, ci_hi=ci_hi, N=N, K=K)

    if vce_type == "cluster":
        ret_dict['_df_r'] = t_df
        ret_dict['g'] = g

    return ret_dict


def robust_vce(vce_type, xpx_inv, x, resid, n, k, cluster=None, g=None,
               spatial_x=None, spatial_y=None, spatial_band=None,
               spatial_kern=None, cols=None):
    """
    Robust variance estimators.
    """
    if cols is None:
        cols = x.columns

    # Homoskedastic
    if vce_type is None:
        s2 = np.dot(resid, resid) / (n - k)
        Sigma = s2 * xpx_inv
        return _wrapSigma(Sigma, cols)
    else:
        xu = x.mul(resid, axis=0).values

    # Robust
    if vce_type in ('hc1', 'robust'):
        xu *= np.sqrt(n / (n-k))
    elif vce_type in ('hc2', 'hc3'):
        h = _get_h(x, xpx_inv)[:, np.newaxis]
        if vce_type == 'hc2':
            xu /= np.sqrt(1 - h)
        elif vce_type == 'hc3':
            xu /= 1 - h
    elif vce_type == 'cluster':
        # sum w/in cluster groups
        int_cluster = pd.factorize(cluster)[0]
        xu = np.array([np.bincount(int_cluster, weights=xu[:, col])
                       for col in range(xu.shape[1])]).T

        xu *= np.sqrt(((n-1)/(n-k))*(g/(g-1)))
    elif vce_type == 'spatial':
        Wxu = dist_weights(xu, spatial_x, spatial_y, spatial_kern,
                           spatial_band)
        Wxu *= n/(n - k)    # No dof correction? See Conley (2008)
    else:
        raise ValueError("`vce_type` '{}' is invalid".format(vce_type))

    try:
        B = xu.T.dot(Wxu)
    except NameError:
        B = xu.T.dot(xu)

    Sigma = _wrapSigma(xpx_inv.dot(B).dot(xpx_inv.T), cols)

    return Sigma

def _get_h(x, xpx_inv):         #noqa
    n = x.shape[0]
    h = np.zeros(n)
    for i in xrange(n):
        x_row = x.iloc[i, :]
        h[i] = x_row.dot(xpx_inv).dot(x_row)
    return h

def _wrapSigma(Sigma, cols):    #noqa
    return pd.DataFrame(Sigma, index=cols, columns=cols)


class Results(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def summary(self):
        if hasattr(self, 'summary'):
            return self._summary
        else:
            out = pd.concat((self.beta, self.se, self.t_stat, self.pt,
                             self.ci_lo, self.ci_hi), axis=1)
            out.columns = ['coeff', 'se', 't', 'p>t', 'CI_low', 'CI_high']
            self._summary = out
            return self._summary

    @property
    def df_m(self):
        """Degrees of freedom for non-constant parameters"""
        try:
            return self._df_m
        except AttributeError:
            self._df_m = self.K

            if not self._nocons:
                self._df_m -= 1

            return self._df_m

    @df_m.setter
    def df_m(self, value):
        self._df_m = value

    @property
    def df_r(self):
        try:
            return self._df_r
        except AttributeError:
            self._df_r = self.N - self.K
            return self._df_r

    @property
    def ssr(self):
        try:
            return self._ssr
        except AttributeError:
            self._ssr = self.resid.dot(self.resid)
            return self._ssr

    @property
    def sst(self):
        return self._sst

    @sst.setter
    def sst(self, y):
        y_demeaned = y - np.mean(y)
        self._sst = y_demeaned.dot(y_demeaned)

    @property
    def r2(self):
        try:
            return self._r2
        except AttributeError:
            self._r2 = 1 - self.ssr/self.sst
            return self._r2

    @property
    def r2_a(self):
        try:
            return self._r2_a
        except AttributeError:
            self._r2_a = (
                1 - (self.ssr/(self.N - self.K))/(self.sst/(self.N - 1)))
            return self._r2_a

    def Ftest(self, col_names, equal=False):
        cols = force_list(col_names)
        V = self.vce.loc[cols, cols]
        q = len(cols)
        beta = self.beta.loc[cols]

        if equal:
            q -= 1
            R = np.zeros((q, q+1))
            for i in xrange(q):
                R[i, i] = 1
                R[i, i+1] = -1
        else:
            R = np.eye(q)

        r = np.zeros(q)

        return f_stat(V, R, beta, r, self.df_r)

    @property
    def F(self):
        """F-stat for 'are all *slope* coefficients zero?'"""
        try:
            return self._F
        except AttributeError:
            # TODO: What if the constant isn't '_cons'?
            cols = [x for x in self.vce.index if x != '_cons']
            self._F, self._pF = self.Ftest(cols)
            return self._F

    @property
    def pF(self):
        try:
            return self._pF
        except AttributeError:
            __ = self.F  # noqa `F` also sets `pF`
            return self._pF


def f_stat(V, R, beta, r, df_d):
    Rbr = (R.dot(beta) - r)
    if Rbr.ndim == 1:
        Rbr = Rbr.reshape(-1, 1)

    middle = la.inv(R.dot(V).dot(R.T))
    df_n = matrix_rank(R)
    # Can't just squeeze, or we get a 0-d array
    F = (Rbr.T.dot(middle).dot(Rbr)/df_n).flatten()[0]
    pF = 1 - stats.f.cdf(F, df_n, df_d)
    return F, pF


def dist_weights(xu, x, y, kernel, band):
    N, K = xu.shape
    Wxu = np.zeros((N, K))

    xarr = x.squeeze().values.astype(float)
    yarr = y.squeeze().values.astype(float)
    kern_func = dist_kernels(kernel, band)
    for i in xrange(N):
        dist = np.abs(np.sqrt((xarr[i] - xarr)**2 + (yarr[i] - yarr)**2))
        w_i = kern_func(dist).astype(np.float64)
        # non_zero = w_i > 1e-10
        # Wxu_i = np.average(xu[non_zero, :], weights=w_i[non_zero], axis=0)
        Wxu_i = w_i.dot(xu)
        Wxu[i, :] = Wxu_i
    return Wxu


def dist_kernels(kernel, band):

    def unif(x):
        return x <= band

    def tria(x):
        return (1 - x/band)*(x <= band)

    if kernel == 'unif':
        return unif
    elif kernel == 'tria':
        return tria


if __name__ == '__main__':
    from os import path
    test_path = path.split(path.relpath(__file__))[0]
    data_path = path.join(test_path, 'tests', 'data')
    df = pd.read_stata(path.join(data_path, 'auto.dta'))
    if 0 == 1:
        y_name = 'price'
        cluster = 'gear_ratio'
        # ols.fit(y, x, cluster=cluster)
        rhv = ['mpg', 'length']
        results = reg(df, y_name, rhv, cluster=cluster, a_name=cluster)
        print results.summary
    elif 1 == 1:
        y = 'price'
        x = ['mpg', 'length']
        z = ['weight', 'trunk']
        w = []
        cluster = 'gear_ratio'
        tsls = ivreg(df, y, x, z, w, addcons=True, cluster='gear_ratio')
        print tsls.summary
        liml = ivreg(df, y, x, z, w, addcons=True, cluster='gear_ratio',
                     method='liml')
        print liml.summary
