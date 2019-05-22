import pandas as pd
import numpy as np
import numpy.linalg as la    # scipy.linalg yields slightly diff results (tsls)
from scipy.linalg import sqrtm              # notin `numpy.linalg`

import scipy.stats as stats

from econtools.util.gentools import force_list, force_df
from econtools.metrics.regutil import (unpack_shac_args, flag_sample,
                                       flag_nonsingletons, set_sample,)
from econtools.metrics.results import Results


# Workhorse classes
class RegBase(object):

    def __init__(self, df, y_name, x_name, **kwargs):
        self.df = df
        self.y_name = y_name
        self.__dict__.update(kwargs)

        self.sample_cols_labels = (
            'y_name', 'x_name', 'fe_name', 'cluster', 'shac_x', 'shac_y',
            'awt_name'
        )

        self.sample_store_labels = (
            'y', 'x', 'A', 'cluster_id', 'shac_x', 'shac_y', 'AWT'
        )

        self.vars_in_reg = ('y', 'x')
        self.add_constant_to = 'x'

        # Set `vce_type`
        self.vce_type = _set_vce_type(self.vce_type, self.cluster, self.shac)
        # Unpack spatial HAC args
        sp_args = unpack_shac_args(self.shac)
        self.shac_x = sp_args[0]
        self.shac_y = sp_args[1]
        self.shac_band = sp_args[2]
        self.shac_kern = sp_args[3]

        # Force variable names to lists
        self.x_name = force_list(x_name)

    def main(self):
        self.set_sample()
        self.estimate()
        self.get_vce()
        self.set_dof()
        self.inference()
        self.results.pull_metadata(self)

        return self.results

    def set_sample(self):
        sample_cols = tuple(
            [self.__dict__[x] for x in self.sample_cols_labels])
        self.sample = flag_sample(self.df, *sample_cols)
        if self.nosingles and self.fe_name:
            self.sample &= flag_nonsingletons(self.df, self.fe_name,
                                              self.sample)

        sample_vars = set_sample(self.df, self.sample, sample_cols)
        self.__dict__.update(dict(zip(self.sample_store_labels, sample_vars)))
        self.x = force_df(self.x)
        self.y = self.y.squeeze()

        # Force regression variables to float64
        for var in self.vars_in_reg:
            self.__dict__[var] = self.__dict__[var].astype(np.float64)

        # Demean or add constant
        if self.fe_name is not None:
            self._demean_sample()
        elif self.addcons:
            _cons = np.ones(self.y.shape[0])
            x = self.__dict__[self.add_constant_to]
            if x.empty:
                x = pd.DataFrame(_cons, columns=['_cons'], index=self.y.index)
            else:
                x['_cons'] = _cons
            self.__dict__[self.add_constant_to] = x

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
        """ Defined by Implementation. Initializes self.Results object """
        raise NotImplementedError

    def get_vce(self):
        """
        Add estimates of Variance-Covariance matrix (VCE), yhat, and residuals
        to `results`.
        """
        X_inner_sum, X_for_resid = self._prep_inference_mats()

        yhat = np.dot(X_for_resid, self.results.beta)
        resid = self.y - yhat

        # Check through VCE types
        xpx_inv = self.results.xpx_inv
        if self.vce_type is None:
            vce = vce_homosk(xpx_inv, resid)
        elif self.vce_type in ('robust', 'hc1'):
            vce = vce_robust(xpx_inv, resid, X_inner_sum)
        elif self.vce_type in ('hc2', 'hc3'):
            vce = vce_hc23(xpx_inv, resid, X_inner_sum, hctype=self.vce_type)
        elif self.vce_type == 'cluster':
            vce = vce_cluster(xpx_inv, resid, X_inner_sum, self.cluster_id)
        elif self.vce_type == 'shac':
            vce = vce_shac(xpx_inv, resid, X_inner_sum,
                           self.shac_x, self.shac_y, self.shac_kern,
                           self.shac_band)
        else:
            raise ValueError

        # Make sure it's symmetric (floating point error)
        vce = _wrapSigma((vce + vce.T) / 2, X_for_resid.columns)

        self.results._add_stat('vce', vce)
        self.results._add_stat('yhat', yhat)
        self.results._add_stat('resid', resid)

        # Not VCE, but needs to go somewhere
        self.results._add_stat('sample', self.sample)

    def _prep_inference_mats(self):
        """
        Set matrices for Sandwich estimator.

        Note: Keep as separate method for sub-classes to override when
          necessary.
        """
        X_for_inner_sum = self.x
        X_for_residual = self.x
        return X_for_inner_sum, X_for_residual

    def set_dof(self):
        """
        Set degrees of freedom used in hypothesis tests and do DoF correction
        on VCE matrix.
        """
        N, K = self._set_NK()
        vce_type = self.vce_type

        if vce_type in (None, 'robust', 'hc1'):
            df, vce_correct = df_std(N, K)
        elif vce_type in ('hc2', 'hc3'):
            df, vce_correct = df_hc23(N, K)
        elif vce_type == 'cluster':
            df, vce_correct, g = df_cluster(N, K, self.cluster_id)
            self.results._add_stat('g', g)
        elif vce_type == 'shac':
            df, vce_correct = df_shac(N, K)

        self.results._add_stat('N', N)
        self.results._add_stat('K', K)
        self.results._add_stat('df_t', df)
        self.results._add_stat('_df_r', df)
        self.results._vce_correct = vce_correct
        self.results.vce *= vce_correct

    def _set_NK(self):
        """
        Do this in a separate method so `IVReg` can tweak `K`
        """
        # Set `N` and `K`
        N, K = self.x.shape

        if self.A is not None:
            self.fe_count = len(self.A.unique())
            if not _fe_nested_in_cluster(self.cluster_id, self.A):
                K += self.fe_count          # Adjust dof's for group means
            self.results.sst = self.y_raw
            self.results._nocons = True
        else:
            self.results._nocons = self.nocons

        return N, K

    def inference(self):
        vce = self.results.vce
        beta = self.results.beta
        t_df = self.results.df_t

        se = pd.Series(np.sqrt(np.diagonal(vce)), index=vce.columns)
        t_stat = beta.div(se)
        p_values = pd.Series(
            stats.t.cdf(-np.abs(t_stat), t_df)*2,  # `t.cdf` is P(x<X)
            index=vce.columns
        )

        self.results._add_stat('se', se)
        self.results._add_stat('t_stat', t_stat)
        self.results._add_stat('pt', p_values)

        conf_level = .95
        crit_value = stats.t.ppf(conf_level + (1 - conf_level)/2, t_df)
        ci_lo = beta - crit_value*se
        ci_hi = beta + crit_value*se

        self.results._add_stat('ci_lo', ci_lo)
        self.results._add_stat('ci_hi', ci_hi)

def _set_vce_type(vce_type, cluster, shac):
    """ Check for argument conflicts, then set `vce_type` if needed.  """

    if vce_type not in (None, 'robust', 'hc1', 'hc2', 'hc3', 'cluster',
                        'shac'):
        raise ValueError("VCE type '{}' is not supported".format(vce_type))

    # Check for conflicts
    if cluster and shac:
        raise ValueError("Cannot use `cluster` and `shac` together.")
    elif cluster and vce_type != 'cluster' and vce_type is not None:
        raise ValueError("Cannot pass argument to `cluster` and set `vce_type`"
                         " to something other than 'cluster'")
    elif shac and vce_type != 'shac' and vce_type is not None:
        raise ValueError("Cannot pass argument to `shac` and set `vce_type`"
                         " to something other than 'shac'")

    # Set `vce_type`
    new_vce = 'cluster' if cluster else 'shac' if shac else vce_type

    return new_vce

def _demean(A, df):
    """ Demean a matrix/DataFrame within group `A` """
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

def _calc_aweights(aw):
    scaled_total = aw.sum() / len(aw)
    row_weights = np.sqrt(aw / scaled_total)
    return row_weights

def _fe_nested_in_cluster(cluster_id, A):
    """ Check if FE's are nested within clusters (affects DOF correction). """
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

def _wrapSigma(Sigma, cols):
    return pd.DataFrame(Sigma, index=cols, columns=cols)


class Regression(RegBase):

    def __init__(self, *args, **kwargs):
        super(Regression, self).__init__(*args, **kwargs)

    def estimate(self):
        beta, xpx_inv = fitguts(self.y, self.x)
        self.results = Results(beta=beta, xpx_inv=xpx_inv)
        self.results.sst = self.y


class IVReg(RegBase):

    def __init__(self, df, y_name, x_name, z_name, w_name, **kwargs):
        super(IVReg, self).__init__(df, y_name, x_name, **kwargs)
        # Handle extra variable stuff for IV
        self.z_name = force_list(z_name)
        self.w_name = force_list(w_name)
        self.sample_cols_labels += ('z_name', 'w_name')
        self.sample_store_labels += ('z', 'w')
        self.vars_in_reg += ('z', 'w')
        self.add_constant_to = 'w'

    def estimate(self):
        y = self.y
        x = self.x
        w = self.w
        z = self.z

        if self.iv_method == '2sls':
            self.Xhat, self.Xtrue = self._first_stage(x, w, z)
            beta, xpx_inv = fitguts(self.y, self.Xhat)

        elif self.iv_method == 'liml':
            beta, xpx_inv, self.Xhat, self.Xtrue, kappa = self._liml(
                y, x, z, w, self._kappa_debug, self.vce_type
            )

        else:
            raise ValueError(
                "IV method '{}' not supported".format(self.method))

        self.results = Results(beta=beta, xpx_inv=xpx_inv)
        self.results.sst = self.y
        self.results._r2 = np.nan
        self.results._r2_a = np.nan
        self.results._add_stat('iv_method', self.iv_method)
        if self.iv_method == 'liml':
            self.results._add_stat('kappa', kappa)

    def _first_stage(self, x, w, z):
        X = pd.concat((x, w), axis=1)
        Xhat = X.copy()
        Z = pd.concat((z, w), axis=1)
        for an_x in x.columns:
            this_x = x[an_x]
            pi_hat, __ = fitguts(this_x, Z)
            Xhat[an_x] = np.dot(Z, pi_hat)
        return Xhat, X

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

        xpx_inv = la.inv(
            (1-kappa)*XX + kappa*np.dot(XZ.dot(ZZ_inv), XZ.T)
        )
        xpy = (1-kappa)*Xy + kappa*np.dot(XZ.dot(ZZ_inv), Zy)
        beta = pd.Series(xpx_inv.dot(xpy).squeeze(), index=X.columns)

        # LIML uses non-standard 'bread' in the sandwich estimator
        if vce_type is None:
            se_xpx_inv = xpx_inv
        else:
            se_xpx_inv = xpx_inv.dot(XZ).dot(ZZ_inv)

        return beta, se_xpx_inv, Z, X, kappa

    def _liml_kappa(self, y, x, w, Z):
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

    def _prep_inference_mats(self):
        """
        In 2SLS, true X is used to calculate residuals, Xhat used in sandwich
        estimator (because it is used in calculating beta-hat).
        """
        X_for_inner_sum = self.Xhat
        X_for_residual = self.Xtrue
        return X_for_inner_sum, X_for_residual

    def _set_NK(self):
        N, K = super(IVReg, self)._set_NK()
        if self.w is not None:
            K += self.w.shape[1]
        return N, K


def fitguts(y, x):
    """ Checks dimensions, inverts, returns beta estimate and (X'X)^-1 """
    # Y should be 1D
    assert y.ndim == 1
    # X should be 2D
    assert x.ndim == 2

    xpx_inv = la.inv(np.dot(x.T, x))
    xpy = np.dot(x.T, y)
    beta = pd.Series(np.dot(xpx_inv, xpy).squeeze(), index=x.columns)

    return beta, xpx_inv

# VCE estimators
def vce_homosk(xpx_inv, resid):
    """ Standard OLS VCE with spherical errors. """
    s2 = np.dot(resid, resid) / resid.shape[0]
    vce = s2 * xpx_inv
    return vce


def vce_robust(xpx_inv, resid, x):
    xu = x.mul(resid, axis=0).values

    B = xu.T.dot(xu)
    vce = sandwich(xpx_inv, B, xpx_inv.T)
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
    vce = sandwich(xpx_inv, B, xpx_inv.T)
    return vce

def _get_h(x, xpx_inv):
    n = x.shape[0]
    h = np.zeros(n)
    for i in range(n):
        x_row = x.iloc[i, :]
        h[i] = x_row.dot(xpx_inv).dot(x_row)
    return h


def vce_cluster(xpx_inv, resid, x, cluster):
    raw_xu = x.mul(resid, axis=0).values

    int_cluster = pd.factorize(cluster)[0]
    xu = np.array([np.bincount(int_cluster, weights=raw_xu[:, col])
                   for col in range(raw_xu.shape[1])]).T

    B = xu.T.dot(xu)
    vce = sandwich(xpx_inv, B, xpx_inv.T)
    return vce


def vce_shac(xpx_inv, resid, x, shac_x, shac_y, shac_kern, shac_band):
    xu = x.mul(resid, axis=0).values
    Wxu = _shac_weights(xu, shac_x, shac_y, shac_kern, shac_band)

    B = xu.T.dot(Wxu)
    vce = sandwich(xpx_inv, B, xpx_inv.T)
    return vce

def _shac_weights(xu, lon, lat, kernel, band):
    N, K = xu.shape
    Wxu = np.zeros((N, K))

    lon_arr = lon.squeeze().values.astype(float)
    lat_arr = lat.squeeze().values.astype(float)
    kern_func = _shac_kernels(kernel, band)
    for i in range(N):
        dist = np.sqrt((lon_arr[i] - lon_arr)**2 + (lat_arr[i] - lat_arr)**2)
        w_i = kern_func(dist).astype(np.float64)
        Wxu[i, :] = w_i.dot(xu)

    return Wxu

def _shac_kernels(kernel, band):

    def unif(x):
        return x <= band

    def tria(x):
        return (1 - x/band)*(x <= band)

    if kernel == 'unif':
        return unif
    elif kernel == 'tria':
        return tria


def sandwich(left, B, right):
    return left.dot(B).dot(right)


# DOF definitions
def df_std(n, k):
    df = n - k
    vce_correct = n / df
    return df, vce_correct


def df_hc23(n, k):
    df = n - k
    vce_correct = 1
    return df, vce_correct


def df_cluster(n, k, cluster_id):
    g = len(pd.value_counts(cluster_id))
    df = g - 1
    vce_correct = ((n - 1) / (n - k)) * (g / (g - 1))
    return df, vce_correct, g


def df_shac(n, k):
    df = n - k
    vce_correct = 1
    return df, vce_correct
