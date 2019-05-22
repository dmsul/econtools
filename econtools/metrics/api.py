import warnings
from typing import Union, List, Optional

import pandas as pd

from econtools.metrics.core import Regression, IVReg
from econtools.metrics.results import Results

RHV = Union[str, List[str]]


def reg(df: pd.DataFrame,
        y_name: str, x_name: Union[str, List[str]],
        fe_name: Optional[str]=None, a_name: Optional[str]=None,
        nosingles: bool=True,
        vce_type: Optional[str]=None, cluster: Optional[str]=None, shac:
        Optional[dict]=None,
        addcons: Optional[bool]=None, nocons: bool=False,
        awt_name: Optional[str]=None
        ) -> Results:
    """OLS Regression.

    Args:
        df (DataFrame): Data with any relevant variables.
        y_name (str): Column name in ``df`` of the dependent variable.
        x_name (str or list): Column name(s) in ``df`` of the independent
                variables/regressors

    Keyword Args:
        vce_type (str): Type of estimator to use for variance-covariance matrix
            of estimated coefficients. Default is standard OLS. Possible
            choices are:
                - 'robust' or 'hc1'
                - 'hc2'
                - 'hc3'
                - 'cluster' (requires kwarg ``cluster``)
                - 'shac' (requires kwarg ``shac``)
        cluster (str): Column name in ``df`` used to cluster standard errors.
        shac (dict): Arguments to pass to spatial HAC estimator.
            Requires:
                - **x** (*str*): Column name in ``df`` to serve as longitude.
                - **y** (*str*): Column name in ``df`` to serve as latitude.
                - **kern** (*str*): Kernel to use in estimation. May be
                  triangle (``tria``) or uniform (``unif``).
                - **band** (float): Bandwidth for kernel.
        fe_name (str) - Column name in ``df`` that defines groups for within
            transformation (demeaning).
        a_name (str) - Deprecated. See ``fe_name``.
        awt_name (str): Column name in ``df`` to use for analytic weights in
            regression.
        addcons (bool): Defaults to False. Add a constant to independent
            variables. Has no effect if ``a_name`` is passed.
        nocons (bool): Defaults to False. Flag so estimators know that
            independent variables ``df`` do not include a constant. Only
            affects degrees of freedom.
        nosingles (bool): Defaults to True. Drop observations that are obsorbed
            by the within transformation. Has no effect if ``a_name=None``.

    Returns:
        A :py:class:`~econtools.metrics.core.Results` object
    """

    fe_name = _a_name_deprecation_handling(a_name, fe_name)

    RegWorker = Regression(
        df, y_name, x_name,
        fe_name=fe_name, nosingles=nosingles, addcons=addcons, nocons=nocons,
        vce_type=vce_type, cluster=cluster, shac=shac,
        awt_name=awt_name,
    )

    results = RegWorker.main()
    return results


def ivreg(df: pd.DataFrame,
          y_name: str,
          x_name: Union[str, List[str]], z_name: Union[str, List[str]],
          w_name: Union[str, List[str]],
          fe_name: Optional[str]=None, a_name: Optional[str]=None,
          nosingles: bool=True,
          iv_method: str='2sls', _kappa_debug=None,
          vce_type: Optional[str]=None, cluster: Optional[str]=None,
          shac: Optional[dict]=None,
          addcons: Optional[bool]=None, nocons: bool=False,
          awt_name: Optional[str]=None,
          ) -> Results:
    """Instrumental Variables Regression

    Args:
        df (DataFrame): Data with any relevant variables.
        y_name (str): Column name in ``df`` of the dependent variable.
        x_name (str or list): Column name(s) in ``df`` of the endogenous
            regressor(s).
        z_name (str or list): Column name(s) in ``df`` of the excluded
            instrument(s)
        w_name (str or list): Column name(s) in ``df`` of the included
            instruments/exogenous regressors

    Keyword Args:
        fe_name (str) - Column name in ``df`` that defines groups for within
            transformation (demeaning). **All other keyword args in
            :py:func:`~econtools.reg` may also be used.
        iv_method (str): Instrumental variables method to use.
            Options are:
                - ``'2sls'``, two-stage least squares (default)
                - ``'liml'``, limited-information maximum likelihood.

    Returns:
        A :py:class:`~econtools.metrics.core.Results` object with (a) no
        r-squared (``r2`` or ``r2_a`` attributes), and (b) a ``kappa``
        attribute (always 1 if ``iv_method='2sls'``)
    """

    fe_name = _a_name_deprecation_handling(a_name, fe_name)

    IVRegWorker = IVReg(
        df, y_name, x_name, z_name, w_name,
        fe_name=fe_name, nosingles=nosingles, addcons=addcons, nocons=nocons,
        iv_method=iv_method, _kappa_debug=_kappa_debug,
        vce_type=vce_type, cluster=cluster, shac=shac,
        awt_name=awt_name,
    )

    results = IVRegWorker.main()
    return results


def _a_name_deprecation_handling(
        a_name: Union[None, str],
        fe_name: Union[None, str]) -> Union[None, str]:
    """
    Nothing deeper than user-facing `reg` and `ivreg` should ever see
    `a_name` argument
    """
    if a_name is not None:
        warnings.warn(
            "Argument `a_name` is deprecated and will be removed in a future "
            "version. Use `fe_name` instead.",
            FutureWarning, stacklevel=3)

        if fe_name is not None and fe_name != a_name:
            raise ValueError("Use only `fe_name`, not `a_name`.")
        else:
            return a_name
    else:
        return fe_name


if __name__ == '__main__':
    from os import path
    import pandas as pd
    test_path = path.split(path.relpath(__file__))[0]
    data_path = path.join(test_path, 'tests', 'data')
    df = pd.read_stata(path.join(data_path, 'auto.dta'))
    y_name = 'price'
    cluster = 'gear_ratio'
    rhv = ['mpg', 'length']
    results = reg(df, y_name, rhv,
                  cluster=cluster,
                  fe_name=cluster,
                  )
    print(results)
