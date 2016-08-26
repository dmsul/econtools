from __future__ import division

from math import factorial

import numpy as np
import pandas as pd

from econtools.metrics import reg


def llr(y, x, x0=None, h=None, N=None, degree=1, kernel='epan'):
    try:
        assert len(y) == len(x)
    except AssertionError:
        raise ValueError("Vectors `y` and `x` must be same size.")

    # Set model parameters
    kernel_obj = kernel_parser(kernel)
    x0 = _set_x0(x, x0, N)      # TODO passed `x0` overwritten?
    h_val = set_bandwidth(y, x, h, kernel_obj)

    # Loop over domain values
    G = np.zeros(len(x0))
    for i, this_x0 in enumerate(x0):
        G[i] = ghat_of_x(y, x, this_x0, h_val, degree, kernel_obj)
    xG = np.stack((x0, G), axis=-1)

    est_stats = {
        'h': h_val,
        'kernel': kernel_obj.name,
        'degree': degree
    }

    return xG, est_stats

def _set_x0(x, x0, N):
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


def set_bandwidth(y, x, h, kernel):
    if type(h) in (float, int):
        h_val = h
    elif type(h) is str or h is None:
        if h in ('thumb', 'silverman', 'rot') or h is None:
            h_val = silverman(x, kernel)
        elif h in ('cv'):
            pass
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


def cross_validation():
    # Loop over i, then h, so `_make_X` only gets called once per i.
    pass


def ghat_of_x(y, x, x0, h, degree, kernel):
    K = kernel_func(x - x0, h, kernel)
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

def _make_X(x, x0, degree):
    centered_x = x - x0
    X = np.vstack(
        [np.ones(len(x))] +
        [centered_x ** i for i in range(1, degree + 1)]
    ).T
    return X


def kernel_func(u, h, kernel):
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
    x = np.sort(np.random.rand(N))
    e = np.random.normal(size=N)
    y = 4 + 5 * x + 0.1 * (x ** 2) + e
    wut, h = llr(y, x, h=None, degree=1)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(wut[:, 0], wut[:, 1], '-r')
    plt.show()
