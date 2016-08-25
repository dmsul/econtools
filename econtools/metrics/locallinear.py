from __future__ import division

import numpy as np
import pandas as pd

from econtools.metrics import reg


def llr(y, x, x0=None, h=None, N=None, degree=1, kernel='epan'):
    try:
        assert len(y) == len(x)
    except AssertionError:
        raise ValueError("Vectors `y` and `x` must be same size.")

    x0 = _set_x0(x, x0, N)      # TODO check that passed `x0` isn't overwritten
    h = set_h(y, x, h)
    G = np.zeros(len(x0))
    for i, this_x0 in enumerate(x0):
        G[i] = ghat_of_x(y, x, this_x0, h, degree, kern_name=kernel)
    return np.stack((x0, G), axis=-1)

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


def set_h(y, x, h):
    if type(h) is float:
        return h
    elif type(h) is str:
        # Check for 'thumb'/silverman or 'cv'
        pass

def _silverman():
    pass

def _cross_validation():
    # Loop over i, then h, so `_make_X` only gets called once per i.
    pass


def ghat_of_x(y, x, x0, h, degree, kern_name):
    K = kernel_func(x - x0, h=h, name=kern_name)
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


def kernel_func(u, h, name='epan'):
    x = u / h
    if name in ('uniform', 'unif', 'rectangle', 'rect'):
        K = _rectangle(x)
    elif name in ('tria', 'triangle'):
        K = _triangle(x)
    elif name == 'epan':
        K = _epan(x)
    return K / h

def _rectangle(x):
    return (np.abs(x) < 1).astype(int)

def _triangle(x):
    return np.maximum((1 - np.abs(x)), 0)

def _epan(x):
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
    wut = llr(y, x, h=.2, degree=1)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(wut[:, 0], wut[:, 1], '-r')
    plt.show()
