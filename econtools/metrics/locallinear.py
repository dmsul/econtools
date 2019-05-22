from math import factorial
from typing import Union, Optional, Tuple, Callable, cast

import numpy as np
import pandas as pd

from econtools.metrics import reg

ArrayLike = Union[np.ndarray, pd.DataFrame]


def kdensity(x: ArrayLike,
             x0: Optional[Union[float, ArrayLike]]=None,
             N: Optional[int]=None, h: Optional[Union[str, float]]=None,
             wt: Optional[ArrayLike]=None,
             kernel: str='epan') -> Tuple[ArrayLike, np.ndarray, dict]:
    """Kernel density estimation.

    Args:
        x (array-like): Variable over which to estimate density.

    Keyword Args:
        x0 (float or array-like): Default ``None``. Values at which to
            caluculate density. If ``None``, these values will be calculated
            automatically. Default length of `x0` is min([len(x), 50]).
            At least one of ``x0`` and ``N`` must be ``None``.
            ``x0`` may also be a scalar.
        N (int): Default ``None``. Number of ``x0`` values to calculate if
            ``x0`` is not specified. At least one of ``x0`` and ``N`` must be
            ``None``.
        h (str or float): Defaults to None (Silverman's rule of thumb).
            Bandwidth for kernel. May pass a float or any of the following for
            Silverman's rule of thumb: ``'silverman'``, ``'thumb'``, ``'rot'``.
        kernel (str): Type of kernel to be used. Options
            are:
                - ``'epan'``, Epanechnikov (default)
                - ``'unif'``, Uniform
                - ``'tria'``, Triangle
        wt (array-like): Weights. Must be same length as ``x``.

    Returns:
        tuple: A tuple containing
                - **x0** (*float or array*) - Points are which kernel is
                  estimated. If ``x0`` is passed explicitly, this will be the
                  same.
                - **f_hat** (*float or array*) - Estimated kernel density at
                  point(s) ``x0``.
                - **est_stats** (*dict*) - Contains bandwidth and kernel name.
    """

    if wt is not None:
        assert len(wt) == len(x)

    # Check inputs
    if not ((x0 is None or N is None)):
        raise ValueError("Cannot specificy `x0` and `N` at the same time.")

    kernel_obj = kernel_parser(kernel)
    x0 = _set_x0(x, x0, N)      # TODO passed `x0` overwritten?
    h_val = set_bandwidth(None, x, h, None, kernel_obj)

    est_stats = {
        'h': h_val,
        'kernel': kernel_obj.name,
    }

    if hasattr(x0, '__iter__'):
        f_hat = np.array(
            [_kdensity_core(x, this_x0, h_val, kernel_obj, wt=wt)
             for this_x0 in x0]
        )
    else:
        f_hat = _kdensity_core(x, x0, h_val, kernel_obj)

    return x0, f_hat, est_stats

def _kdensity_core(x, x0, h, kernel_obj, wt=None):
    k_vals = kernel_func(x - x0, h, kernel_obj)
    if wt is None:
        f_hat = k_vals.mean()
    else:
        f_hat = wt.dot(k_vals) / wt.sum()

    return f_hat


def llr(y: np.ndarray, x: np.ndarray,
        x0: Optional[ArrayLike]=None,
        N: Optional[int]=None, h: Optional[Union[str, float]]=None,
        degree: int=1, kernel: str='epan', ci: bool=False):
    """Local-linear Regression

    Args:
        y (array): Dependent variable
        x (array): Independent variable

    Keyword Args:
        x0 (float or array-like): Default ``None``. Values at which to
            caluculate regression. If ``None``, these values will be calculated
            automatically. Default length of `x0` is min([len(x), 50]).  At
            least one of ``x0`` and ``N`` must be ``None``.
        N (int): Default ``None``. Number of ``x0`` values to calculate if
            ``x0`` is not specified. At least one of ``x0`` and ``N`` must be
            ``None``.
        h (str or float): Defaults to None (Silverman's rule of thumb).
            Bandwidth for kernel. May pass a float or any of the following for
            Silverman's rule of thumb: ``'silverman'``, ``'thumb'``, ``'rot'``.
        kernel (str): Type of kernel to be used. Options
            are:
                - ``'epan'``, Epanechnikov (default)
                - ``'unif'``, Uniform
                - ``'tria'``, Triangle
        degree (int): Defaults to 1. Degree of polynomial to use in local
            regression.
        ci (bool): Defaults to False. If True, also return confidence interval
            for each point.

    Returns:
        Stuff.
    """
    try:
        assert len(y) == len(x)
    except AssertionError:
        raise ValueError("Vectors `y` and `x` must be same size.")

    # Set model parameters
    kernel_obj = kernel_parser(kernel)
    x0 = _set_x0(x, x0, N)      # TODO passed `x0` overwritten?

    if type(h) is str:
        return_only = 'return' in cast(str, h)
    else:
        return_only = False
    h_val = set_bandwidth(y, x, h, degree, kernel_obj, return_only=return_only)
    if return_only:
        return h_val

    # Loop over domain values
    xG = model_loop(y, x, x0, h_val, degree, kernel_obj)

    est_stats = {
        'h': h_val,
        'kernel': kernel_obj.name,
        'degree': degree
    }

    if ci:
        raise NotImplementedError   # TODO, see `NonParametrics2`, sec 3.11
    else:
        return xG, est_stats        # TODO, separate x0 and yhat

def _set_x0(x, x0, N) -> ArrayLike:
    if x0 is not None:
        return x0
    N = _set_N(x, N)
    x_min = np.min(x)
    x_max = np.max(x)
    return np.linspace(x_min, x_max, num=N)

def _set_N(x, N):
    if N is None:
        return np.min([len(x), 50])
    else:
        return N


def model_loop(y, x, x0, h, degree, kernel):
    G = np.zeros(len(x0))
    for i, this_x0 in enumerate(x0):
        G[i] = ghat_of_x(y, x, this_x0, h, degree, kernel)
    xG = np.stack((x0, G), axis=-1)
    return xG


def set_bandwidth(y, x, h, degree, kernel, return_only=False):
    if type(h) in (float, int):
        h_val = h
    elif type(h) is str or h is None:
        if h in ('thumb', 'silverman', 'rot') or h is None:
            h_val = silverman(x, kernel)
        elif 'cv' in h:
            h_set, CV_h = cross_validation(y, x, degree, kernel)
            if return_only:
                return h_set, CV_h
            else:
                argmin = np.argmin(CV_h)
                if argmin in (0, len(CV_h) - 1):
                    raise ValueError("Bandwidth at edge of range")
                h_val = h_set[argmin]
        elif h in ('cv-return'):
            pass
    else:
        raise ValueError

    return h_val

def silverman(x, kernel):
    v = kernel.degree
    roughness = kernel.Rk
    kernel_moment = kernel.kappa
    C_v_of_k = 2 * (
        (np.sqrt(np.pi) * factorial(v) ** 3 * roughness) /
        (2 * v * factorial(2 * v) * kernel_moment)
    ) ** (1 / (2 * v + 1))
    sigma = np.sqrt(np.var(x))
    n = len(x)
    h = sigma * C_v_of_k / n ** (1 / (2 * v + 1))
    return h

def cross_validation(y, x, degree, kernel):
    h_set = _set_hset_for_cv(x, kernel)
    CV_h = np.zeros(len(h_set))
    for h_idx, h in enumerate(h_set):
        this_CV_h = 0
        for xi_idx, x_i in enumerate(x):
            y_no_i = np.delete(y, xi_idx)
            x_no_i = np.delete(x, xi_idx)
            y_hat = ghat_of_x(y_no_i, x_no_i, x_i, h, degree, kernel)
            resid2 = (y[xi_idx] - y_hat) ** 2
            this_CV_h += resid2
        CV_h[h_idx] = this_CV_h
    return h_set, CV_h

def _set_hset_for_cv(x, kernel):
    hs = silverman(x, kernel)
    h_set = np.linspace(.2 * hs, 2 * hs, num=50)
    return h_set


def ghat_of_x(y, x, x0, h, degree, kernel):
    K = kernel_func(x - x0, h, kernel)
    if _sparse_data(K, degree):
        return np.nan

    X = _make_X(x, x0, degree)
    x_name = ['cons'] + ['x{}'.format(i) for i in range(1, degree + 1)]
    df = pd.DataFrame(
        np.hstack((
            y.reshape(-1, 1),
            X,
            K.reshape(-1, 1)
        )),
        columns=['y'] + x_name + ['k']
    )
    res = reg(df, 'y', x_name, awt_name='k')
    beta = res.beta
    # plot_this(y, x, K, X, res)    # XXX tmp, diagnostic
    return beta['cons']

def _sparse_data(K, degree):
    count_nonzero = (K != 0).sum()
    return count_nonzero < degree * 2

def _make_X(x, x0, degree):
    centered_x = x - x0
    X = np.vstack(
        [np.ones(len(x))] +
        [centered_x ** i for i in range(1, degree + 1)]
    ).T
    return X


def kernel_func(u: ArrayLike, h: float, kernel: Callable) -> np.ndarray:
    x = u / h
    K = kernel(x)
    return K / h

def kernel_parser(name):
    if name in ('unif', 'uniform', 'rectangle', 'rect'):
        kern = Uniform_Kernel()
    elif name in ('tria', 'triangle'):
        kern = Triangle_Kernel()
    elif name == 'epan':
        kern = Epanechnikov_Kernel()

    return kern

class Uniform_Kernel(object):

    degree = 2      # Degree (of first non-zero moment)
    kappa = 1 / 3   # First non-zero moment
    Rk = 1 / 2      # Roughness
    name = 'unif'

    def __init__(self):
        pass

    def kernel(self, x):
        return (np.abs(x) < 1).astype(int)

    def __call__(self, x):
        return self.kernel(x)

class Triangle_Kernel(object):

    degree = 2      # Degree (of first non-zero moment)
    kappa = 1 / 6   # First non-zero moment
    Rk = 2 / 3      # Roughness
    name = 'tria'

    def __init__(self):
        pass

    def __call__(self, x):
        return self.kernel(x)

    def kernel(self, x):
        return np.maximum((1 - np.abs(x)), 0)

class Epanechnikov_Kernel(object):

    degree = 2      # Degree (of first non-zero moment)
    kappa = 1 / 5   # First non-zero moment
    Rk = 3 / 5      # Roughness
    name = 'epan'

    def __init__(self):
        pass

    def __call__(self, x):
        return self.kernel(x)

    def kernel(self, x):
        return np.maximum((1 - x ** 2)*(3 / 4), 0)


def plot_this(y, x, K, X, res):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    yhat = X.dot(res.beta)
    ax.plot(x[K > 0], yhat[K > 0], '-r')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(seed=12345)
    N = 1000
    x = np.sort(np.random.rand(N)) * 8
    e = np.random.normal(size=N) * 7
    y = 4 - 5 * x - .4 * (x ** 2) + 0.2 * (x ** 3) + e
    wut, h = llr(y, x, h='rot', degree=1)
    fig, ax = plt.subplots()
    if 1:
        ax.scatter(x, y)
        ax.plot(wut[:, 0], wut[:, 1], '-r')
        print(h)
    else:
        ax.plot(wut, h)
    plt.show()
