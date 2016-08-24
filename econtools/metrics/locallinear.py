from __future__ import division

import numpy as np
import pandas as pd

from econtools.metrics import reg


def llr(y, x, h, N=None, degree=1):
    assert len(y) == len(x)
    N = _set_N(x, N)
    x0 = _set_x0(x, N)
    G = np.zeros(N)
    for i, this_x0 in enumerate(x0):
        G[i] = _actual_reg(y, x, this_x0, h, degree)
    return np.stack((x0, G), axis=-1)


def _set_N(x, N):
    if N is None:
        return np.min([len(x), 50])
    else:
        return N


def _set_x0(x, N):
    x_min = np.min(x)
    x_max = np.max(x)
    return np.linspace(x_min, x_max, num=N)


def _actual_reg(y, x, x0, h, degree):
    K = kernel(x - x0, h=h)
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
    # plot_this(y, x, K, X, res)
    return beta['cons']


def _make_X(x, x0, degree):
    X = np.vstack(
        [np.ones(len(x))] +
        [(x - x0) ** i for i in range(1, degree + 1)]
    ).T
    return X


def kernel(u, h):
    x = u / h
    K = (np.abs(x) < 1).astype(int)
    # K = np.maximum((1 - np.abs(x)), 0)
    # K = np.maximum((1 - x ** 2)*(3 / 4), 0)
    return K / h


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
    wut = llr(y, x, .2, degree=1)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(wut[:, 0], wut[:, 1], '-r')
    plt.show()
