from __future__ import division

import pandas as pd
import numpy as np
import numpy.linalg as la    # scipy.linalg yields slightly diff results (tsls)
from numpy.linalg import matrix_rank        # not in `scipy.linalg`
from scipy.linalg import sqrtm              # notin `numpy.linalg`

import scipy.stats as stats

from econtools.util import force_list, force_df
from regutils import (unpack_spatialargs, flag_sample, flag_nonsingletons,
                      set_sample,
                      )


def reg(df, y_name, x_name,
        a_name=None, nosingles=True,
        vce_type=None, cluster=None, spatial_hac=None,
        addcons=None, nocons=False,
        awt_name=None
        ):
    pass


def ivreg(df, y_name, x_name, z_name, w_name,
          a_name=None, nosingles=True,
          iv_method='2sls', _kappa_debug=None,
          vce_type=None, cluster=None, spatial_hac=None,
          addcons=None, nocons=False,
          awt_name=None,
          ):
    pass


# Workhorse
class RegBase(object):

    def __init__(self, df, y_name, x_name, **kwargs):
        self.df = df
        self.y_name = y_name
        self.__dict__.update(kwargs)

        self.sample_cols_labels = (
            'y_name', 'x_name', 'a_name', 'cluster', 'spatial_x', 'spatial_y',
            'awt_name'
        )

        self.sample_store_labels = (
            'y', 'x', 'A', 'cluster_id', 'space_x', 'space_y', 'AWT'
        )

        self.vars_in_reg = ('y', 'x')
        self.add_constant_to = 'x'

        # Set `vce_type`
        self.vce_type = _set_vce_type(self.vce_type, self.cluster,
                                      self.spatial_hac)
        # Unpack spatial HAC args
        sp_args = unpack_spatialargs(self.spatial_hac)
        self.spatial_x = sp_args[0]
        self.spatial_y = sp_args[1]
        self.spatial_band = sp_args[2]
        self.spatial_kern = sp_args[3]

        # Force variable names to lists
        self.x_name = force_list(x_name)

    def main(self):
        self.set_sample()
        self.estimate()
        self.set_dofs()
        self.get_vce()

    def set_sample(self):
        sample_cols = tuple([self.__dict__[x] for x in self.sample_cols_labels])
        self.sample = flag_sample(self.df, *sample_cols)
        if self.nosingles and self.a_name:
            self.sample &= flag_nonsingletons(self.df, self.a_name, self.sample)

        sample_vars = set_sample(self.df, self.sample, sample_cols)
        self.__dict__.update(dict(zip(self.sample_store_labels, sample_vars)))
        self.x = self.x.to_frame()

        # Demean or add constant
        if self.a_name is not None:
            self._demean_sample()
        elif self.addcons:
            self.__dict__[self.add_constant_to]['_cons'] = np.ones(
                self.x.shape[0])

        # Re-weight sample
        if self.AWT is not None:
            self._weight_sample()

    def _demean_sample(self):
        self.y_raw = self.y.copy()
        for var in self.vars_in_reg:
            self.__dict__[var] = _demean(self.A, self.__dict__[var])

    def _weight_sample(self):
        row_wt = _calc_aweights(self.AWT)
        for var in self.vars_in_reg:
            self.__dict__[var] = self.__dict__[var].multiply(row_wt, axis=0)

    def estimate(self):
        """Defined by Implementation"""
        raise NotImplementedError

    def set_dofs(self):
        N, K = self.x.shape

        if self.A is not None:
            if not _fe_nested_in_cluster(self.cluster_id, self.A):
                K += len(self.A.unique())    # Adjust dof's for group means
            self.results.sst = self.y_raw
            self.results._nocons = True
        else:
            self.results._nocons = self.nocons

        self.N, self.K = N, K

    def get_vce(self):
        self._prep_inference_mats()
        # Check through VCE types
        if self.vce_type is None:
            vce = vce_homosk(self.results.xpx_inv, self.resid)
        # XXX start here

        self.vce = vce

    def _prep_inference_mats(self):
        self.yhat = np.dot(self.x, self.results.beta)
        self.resid = self.y - self.yhat
        # For sandwich estimator
        # XXX Make sure `self.x` doesn't get over-written by sub-classes
        self.Xinner = self.x


class Regression(RegBase):

    def __init__(self, *args, **kwargs):
        super(Regression, self).__init__(*args, **kwargs)

    def estimate(self):
        self.results = fitguts(self.y, self.x)


class IVReg(RegBase):

    def __init__(self, df, y_name, x_name, z_name, w_name, **kwargs):
        super(Regression, self).__init__(df, y_name, x_name, **kwargs)
        # Handle extra variable stuff for IV
        self.z_name = force_list(z_name)
        self.w_name = force_list(w_name)
        self.sample_cols_labels += ('z_name', 'w_name')
        self.sample_store_labels += ('z', 'w')
        self.vars_in_reg += ('z', 'w')
        self.add_constant_to = 'w'

    def estimate(self):
        x = self.x
        w = self.w
        z = self.z
        if self.method == '2sls':
            self.Xhat = self._first_stage(x, w, z)
            self.results = fitguts(self.y, self.Xhat)
        elif self.method == 'liml':
            self.results, self.Xhat, kappa = self._liml(
                self.y, x, z, w, self._kappa_debug, self.vce_type
            )
            self.results.kappa = kappa
        else:
            raise ValueError("IV method '{}' not supported".format(self.method))

        self.results._r2 = np.nan
        self.results._r2_a = np.nan

    def _first_stage(self, x, w, z):
        X_hat = pd.concat((x, w), axis=1)
        Z = pd.concat((z, w), axis=1)
        for an_x in x.columns:
            this_x = x[an_x]
            first_stage = fitguts(this_x, Z)
            X_hat[an_x] = np.dot(Z, first_stage.beta)
        return X_hat

    def _liml(self, y, x, z, w, _kappa_debug, vce_type):
        Z = pd.concat((z, w), axis=1)
        kappa, ZZ_inv = self._liml_kappa(y, x, w, Z)
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

    def _liml_kappa(self, y, x, w, Z):        #noqa
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

    def set_dofs(self):
        super(IVReg, self).set_dofs()
        if self.w is not None:
            self.K += self.w.shape[1]

    def _prep_inference_mats(self):
        super(IVReg, self)._prep_inference_mats()
        self.Xinner = pd.concat((self.x, self.w), axis=1)


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


def _calc_aweights(aw):
    scaled_total = aw.sum() / len(aw)
    row_weights = np.sqrt(aw / scaled_total)
    return row_weights


def _demean(A, df):
    # Ignore empty `df` (e.g. empty list of exogenous included regressors)
    if df is None or df.empty:
        return df
    else:
        group_name = A.name
        mean = df.groupby(A).mean()
        large_mean = force_df(A).join(mean, on=group_name).drop(group_name,
                                                                axis=1)
        if df.ndim == 1:
            large_mean = large_mean.squeeze()
        demeaned = df - large_mean
        return demeaned


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


def _fe_nested_in_cluster(cluster_id, A):
    """
    Check if FE's are nested within clusters (affects DOF correction).
    """
    if (cluster_id is None) or (A is None):
        return False
    elif (cluster_id.name == A.name):
        return True
    else:
        joint = pd.concat((cluster_id, A), axis=1)
        names = [cluster_id.name, A.name]
        pair_counts = joint.groupby(names)[A.name].count()
        num_of_clusters = pair_counts.groupby(level=A.name).count()
        return num_of_clusters.max() == 1


# Results class
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


# VCE estimators
def vce_homosk(xpx_inv, resid):
    """ Standard OLS VCE with spherical errors. """
    s2 = np.dot(resid, resid) / resid.shape[0]
    vce = s2 * xpx_inv
    return vce


def vce_robust(xpx_inv, resid, x):
    xu = x.mul(resid, axis=0).values

    B = xu.T.dot(xu)
    vce = xpx_inv.dot(B).dot(xpx_inv)
    return vce


def vce_hc23(xpx_inv, resid, x, hctype='hc2'):
    xu = x.mul(resid, axis=0).values
    h = _get_h(x, xpx_inv)[:, np.newaxis]
    if hctype == 'hc2':
        xu /= np.sqrt(1 - h)
    elif hctype == 'hc3':
        xu /= 1 - h
    else:
        raise ValueError

    B = xu.T.dot(xu)
    vce = xpx_inv.dot(B).dot(xpx_inv)
    return vce

def _get_h(x, xpx_inv):         #noqa
    n = x.shape[0]
    h = np.zeros(n)
    for i in xrange(n):
        x_row = x.iloc[i, :]
        h[i] = x_row.dot(xpx_inv).dot(x_row)
    return h


def vce_cluster(xpx_inv, resid, x, cluster):
    raw_xu = x.mul(resid, axis=0).values

    int_cluster = pd.factorize(cluster)[0]
    xu = np.array([np.bincount(int_cluster, weights=raw_xu[:, col])
                   for col in range(raw_xu.shape[1])]).T

    B = xu.T.dot(xu)
    vce = xpx_inv.dot(B).dot(xpx_inv)
    return vce


def vce_shac(xpx_inv, resid, x, shac_x, shac_y, shac_kern, shac_band):
    xu = x.mul(resid, axis=0).values
    Wxu = _shac_weights(xu, shac_x, shac_y, shac_kern, shac_band)

    B = xu.T.dot(Wxu)
    vce = xpx_inv.dot(B).dot(xpx_inv)
    return vce

def _shac_weights(xu, lon, lat, kernel, band):      #noqa
    N, K = xu.shape
    Wxu = np.zeros((N, K))

    lon_arr = lon.squeeze().values.astype(float)
    lat_arr = lat.squeeze().values.astype(float)
    kern_func = _shac_kernels(kernel, band)
    for i in xrange(N):
        dist = np.abs(
            np.sqrt(
                (lon_arr[i] - lon_arr)**2 + (lat_arr[i] - lat_arr)**2
            )
        )
        w_i = kern_func(dist).astype(np.float64)
        # non_zero = w_i > 1e-10
        # Wxu_i = np.average(xu[non_zero, :], weights=w_i[non_zero], axis=0)
        Wxu_i = w_i.dot(xu)
        Wxu[i, :] = Wxu_i

    return Wxu

def _shac_kernels(kernel, band):                    #noqa

    def unif(x):
        return x <= band

    def tria(x):
        return (1 - x/band)*(x <= band)

    if kernel == 'unif':
        return unif
    elif kernel == 'tria':
        return tria


if __name__ == '__main__':
    pass
