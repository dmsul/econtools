from __future__ import division

import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
import scipy.linalg as la
import matplotlib.pyplot as plt

from econtools.metrics.locallinear import llr


def kriging_weights(X, y, X0, model_name='exp'):
    model = model_factory(model_name)
    mle, D = variogram_mle(X, y, model)
    param_est = mle['x']

    # Construct K (monitor covariances w/ lagrange multiplier)
    M = len(X)      # Number of monitors
    K = np.ones((M + 1, M + 1))     # Add row for Lagrange Multiplier
    K[:M, :M] = model(D, param_est)
    K[-1, -1] = 0                   # Lagrange
    # Construct k (monitor-target covariance(s), w/ lagrange multiplier)
    N = len(X0)     # Number of interpolation targets
    k = np.ones((M + 1, N))
    dist_to_x0 = cdist(X, X0)
    k[:-1, :] = model(dist_to_x0, param_est)

    # Krige it
    weights = la.inv(K).dot(k)[:-1]     # Drop Lagrange multiplier (idx = -1)

    return weights


def check_variogram(X, y, maxd=None, scat=False,
                    npreg_args=dict(),
                    model_name='exp',
                    mle_args=dict(),
                    ):
    """
    Scatter p = (h_ij, (y_i - y_j) ^ 2).
    Kernel regression of p.
    MLE of variogram model to fit p.
    """

    # Estimate the MLE
    model = model_factory(model_name)
    mle, __ = variogram_mle(X, y, model)
    est = mle['x']
    print mle

    # Get empirical variogram data
    xG, h, sqdiff, __ = llr_gamma(X, y, maxd=maxd, plot=False, ret_raw=True)

    # Plot everything
    x0 = xG[:, 0]                               # Kernel reg x's
    llr = xG[:, 1]                              # Kernel reg y's
    g_model = est[0] - model(x0, est)           # Model fit y's
    fig, ax = plt.subplots()
    ax.scatter(h, sqdiff)                       # Scatter actual (y, sqdiffs)
    ax.plot(x0, llr, '-og')                     # Plot kernel reg
    ax.plot(x0, g_model, '-b')                  # Plot model fit
    plt.show()


def variogram_mle(X, y, model):
    """ Driver for MLE. """
    D = cdist(X, X)
    y_demeaned = y - y.mean()
    N = len(y)
    mle = minimize(_likelihood, [4274, 4], args=(D, y_demeaned, N, model),
                   method='BFGS')
    return mle, D


def model_factory():
    pass


def _likelihood(ca, D, y_demeaned, n, model):
    R = model(D, ca)
    R_inv = la.inv(R)
    L = np.log(la.det(R)) + y_demeaned.dot(R_inv).dot(y_demeaned)
    return L


def llr_gamma(X, y, maxd=None, scat=False, plot=False, ret_raw=False,
              **npregargs):
    """ Estimate variogram using local linear regression. """
    dist, sqdiff = empirical_gamma(X, y, maxd=maxd)
    # plot w/ kernel reg
    xG, est_stats = llr(sqdiff, dist, **npregargs)

    if plot:
        fig, ax = plt.subplots()
        if scat:
            ax.scatter(dist, sqdiff)
        ax.plot(xG[:, 0], xG[:, 1], '-og')
        plt.show()

    if ret_raw:
        return xG, dist, sqdiff, est_stats
    else:
        return xG, est_stats


def empirical_gamma(X, y, maxd=None):
    """
    Raw (distance, squared difference) for every pair in X.

    args
    ----
    X (array) - N x 2 array with arbitrary (x, y) coordinates.
    y (iterable, array-like) - Variable at locations in `X`.

    kwargs
    ------
    maxd (float) - Any distance pairs beyond `maxd` are dropped.

    Returns
    ------
    dist (array) - Flattened upper-trianglular distance matrix of rows in
        `X`. See `scipy.spatial.distance.pdist`.
    sqdiff (array) - Accompanying squared difference in y,
        i.e. (y[i] - y[j]) ^ 2.
    """
    # Calc distances
    dist = pdist(X)
    # Calc squared diff of y
    N = len(X)
    sqdiff = np.zeros(len(dist))
    for i in range(N):
        y_i = y[i]
        for j in range(i + 1, N):
            sqdiff[get_flat_matrix_idx(i, j, N)] = (y_i - y[j]) ** 2

    if maxd:
        sqdiff = sqdiff[dist < maxd]
        dist = dist[dist < maxd]

    return dist, sqdiff


def get_flat_matrix_idx(i, j, n):
    """
    Convert (i, j) indices of matrix to index of flattened upper-triangular
    vector.
    """
    return n*i - i*(i + 1) // 2 + j - i - 1


if __name__ == "__main__":
    pass
