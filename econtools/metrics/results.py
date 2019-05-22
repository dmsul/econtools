from typing import Tuple

import pandas as pd
import numpy as np
import numpy.linalg as la    # scipy.linalg yields slightly diff results (tsls)
from numpy.linalg import matrix_rank        # not in `scipy.linalg`
import scipy.stats as stats

from econtools.util.gentools import force_list


class Results(object):
    """Regression Results container.

    Attributes:
        summary (DataFrame): Summary of regression results.
        beta (Series): All beta coefficients. Index is regressor names.
        se (Series): Standard errors.
        t_stat (Series): t-stats.
        pt (Series): p-scores for t-stats.
        ci_lo (Series): Confidence interval, lower bound.
        ci_hi (Series): Confidence interval, upper bound.
        r2 (float): R-squared
        r2_a (float): Adjusted R-squared.
        K (int): Number of regressors
        N (int): Number of observations
        vce (DataFrame): K-by-K variance-covariance matrix.
        F (float): F-stat of joint significance of beta coefficients.
        pF (float): p-score for F-stat.
        df_m (int): Model degrees of freedom (excluding constant).
        df_r (int): Residual degrees of freedom.
        ssr (float): Sum of squared residuals.
        sst (float): Total sum of squares.
        yhat (array): Fit values (:math:`X\\hat{\\beta}`)
        resid (array): Regression residuals (:math:`\\hat{\\varepsilon}`)
        sample (array): Boolean array the same length of DataFrame passed to
            original regression function. Row is `True` is the observation is
            included in the regression, `False` otherwise. Regression function
            will automatically drop observations where the outcome, regressor,
            weights, etc., are missing/null.
    """

    def __init__(self, **kwargs):
        self.beta = None    # For mypy
        self.se = None
        self.__dict__.update(kwargs)

    def pull_metadata(self, Reg):
        # Do NOT store full `reg_object` (contains source data);
        # only keep meta data/config info
        self.y_name = Reg.y_name
        self.x_name = Reg.x_name
        self.vce_type = Reg.vce_type
        if Reg.fe_name:
            self.fe_name = Reg.fe_name
            self.fe_count = Reg.fe_count
        if Reg.cluster:
            self.cluster_name = Reg.cluster
        if Reg.shac_kern:
            self.shac_kernel = Reg.shac_kern
            self.shac_bandwidth = Reg.shac_band

    def __repr__(self):
        border_str = '='*55 + '\n'
        out_str = border_str
        out_str += f'Dependent variable:\t{self.y_name}\n'
        out_str += f'N:\t\t\t{self.N}\n'
        out_str += f'R-squared:\t\t{self.r2:.4f}\n'

        out_str += 'Estimation method:\t'
        if hasattr(self, 'iv_method'):
            out_str += f'{self.iv_method.upper()}\n'
            if self.iv_method == 'liml':
                out_str += f'  Kappa:\t\t  {self.kappa}\n'
        else:
            out_str += 'OLS\n'

        out_str += 'VCE method:\t\t'
        if self.vce_type is None:
            out_str += 'Standard (Homosk.)\n'
        else:
            vce = self.vce_type
            vce = vce.upper() if len(vce) < 5 else vce.title()
            out_str += f'{vce}\n'
        if hasattr(self, 'cluster_name'):
            out_str += f'  Cluster variable:\t  {self.cluster_name}\n'
            out_str += f'  No. of clusters:\t  {self.g}\n'
        elif hasattr(self, 'shac_kernel'):
            out_str += f'  SHAC kernel:\t  {self.shac_kernel}\n'
            out_str += f'  SHAC bandwidth:\t  {self.shac_bandwidth}\n'

        if hasattr(self, 'fe_name'):
            out_str += f'Fixed effects by:\t{self.fe_name}\n'
            out_str += f'  No. of FE:\t\t  {self.fe_count}\n'

        out_str += border_str

        out_str += self.summary.to_string(
            formatters=[
                lambda x: f'{x:.3f}',       # Coeff
                lambda x: f'{x:.3f}',       # std. err.
                lambda x: f'{x:.3f}',       # t-stat
                lambda x: f'{x:.3f}',       # p-score
                lambda x: f'{x:.3f}',       # CI-low
                lambda x: f'{x:.3f}',       # CI-high
            ]
        ) + '\n'

        out_str += border_str

        return out_str

    # TODO: Why do I wrap this in a method? Why does `_add_stat` exist?
    # Ans: I think to automate setting w/o using __dict__ every time?
    def _add_stat(self, stat_name, stat):
        self.__dict__[stat_name] = stat

    @property
    def summary(self):
        if hasattr(self, '_summary'):
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
        """F test using regression results.

        Args:
            col_names (str or list): Regressor name(s) to test.

        Keyword Args:
            equal (bool): Defaults to False. If True, test if all coefficients
                in ``col_names`` are equal. If False, test if ``col_names`` are
                jointly significant.

        Returns:
            tuple: A tuple containing:
                - **F** (float): F-stat.
                - **pF** (float): p-score for ``F``.
        """
        cols = force_list(col_names)
        V = self.vce.loc[cols, cols]
        q = len(cols)
        beta = self.beta.loc[cols]

        if equal:
            q -= 1
            R = np.zeros((q, q+1))
            for i in range(q):
                R[i, i] = 1
                R[i, i+1] = -1
        else:
            R = np.eye(q)

        r = np.zeros(q)

        return f_test(V, R, beta, r, self.df_r)

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
            self.F  # `F` also sets `pF`
            return self._pF


# TODO: Roll this into Results object?
def f_test(V: np.ndarray, R: np.ndarray, beta: np.ndarray, r: int,
           df_d: int) -> Tuple[float, float]:
    """Arbitrary F test.

    Args:
        V (array): K-by-K variance-covariance matrix.
        R (array): K-by-K Test matrix.
        beta (array): Length-K vector of coefficient estimates.
        r (array): Length-K vector of null hypotheses.
        df_d (int): Denominator degrees of freedom.

    Returns:
        tuple: A tuple containing:
            - **F** (float): F-stat.
            - **pF** (float): p-score for ``F``.
    """
    Rbr = (R.dot(beta) - r)
    if Rbr.ndim == 1:
        Rbr = Rbr.reshape(-1, 1)

    middle = la.inv(R.dot(V).dot(R.T))
    df_n = matrix_rank(R)
    # Can't just squeeze, or we get a 0-d array
    F = (Rbr.T.dot(middle).dot(Rbr)/df_n).flatten()[0]
    pF = 1 - stats.f.cdf(F, df_n, df_d)
    return F, pF
