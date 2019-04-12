import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
import scipy.linalg as la
import matplotlib.pyplot as plt

from econtools.metrics.locallinear import llr


def kriging_weights(X, y, X0, model_name='exp', mle_args=dict()):
    """
    Calculate Simple Kriging weights for monitor readings (`X`, `y`) at
    locations `X0` using variogram model `model_name`.

    Args
    ----
    X (array, dim M x 2) - Monitor locations
    y (array, len M) - Spatial variable values
    X0 (array, dim N x 2) - Interpolation targets

    Kwargs
    ------
    model_name (str) - Class of model to use for variogram estimation. Valid
        values are
            - 'exp' (Default)
            - 'gauss'
            - 'spherical'
        See documentation for details.

    mle_args (dict) - Pass following to MLE estimation routine:
        `param0` (iterable) - Initial values for `model_name`
        `method` (str) - Optimization method (see `scipy.optimize.minimum`)


    Returns
    -------
    weights (array, dim N x M) - Kriging weights
    """

    model = model_factory(model_name)
    mle, D = variogram_mle(X, y, model, mle_args)
    param_est = mle['x']

    # Construct K (monitor covariances w/ lagrange multiplier)
    M = len(X)                              # Number of monitors
    K = np.ones((M + 1, M + 1))             # Add row for Lagrange Multiplier
    K[:M, :M] = model(D, param_est)
    K[-1, -1] = 0                           # For Lagrange
    # Construct k (monitor-target covariance(s), w/ Lagrange multiplier)
    N = len(X0)                             # Number of interpolation targets
    k = np.ones((M + 1, N))
    dist_to_x0 = cdist(X, X0)
    k[:-1, :] = model(dist_to_x0, param_est)    # Assign around Lagrange

    # Krige it
    weights = la.inv(K).dot(k)[:-1].T           # Drop Lagrange

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
    mle, __ = variogram_mle(X, y, model, mle_args)
    est = mle['x']
    print(mle)

    # Get empirical variogram data
    xG, h, sqdiff, est_stats = llr_gamma(X, y, maxd=maxd, plot=False,
                                         ret_raw=True)

    # Plot everything
    x0 = xG[:, 0]                               # Kernel reg x's
    llr = xG[:, 1]                              # Kernel reg y's
    fig, ax = plt.subplots()
    if scat:
        ax.scatter(h, sqdiff)                   # Scatter actual (y, sqdiffs)
    ax.plot(x0, llr, '-og')                     # Plot kernel reg
    fullx = np.linspace(0, x0.max(), 100)
    g_model = est[0] - model(fullx, est)           # Model fit y's
    ax.plot(fullx, g_model, '-b')                  # Plot model fit
    plt.show()

    return xG, h, sqdiff, est_stats


# MLE Driver
def variogram_mle(X, y, model, mle_args):
    """ Driver for MLE. """
    D = cdist(X, X)
    y_demeaned = y - y.mean()
    N = len(y)
    # Extract MLE args
    param0 = mle_args.get('param0')
    method = mle_args.get('method', 'BFGS')
    # Estimate
    mle = minimize(_likelihood, param0, args=(D, y_demeaned, N, model),
                   method=method)
    return mle, D

def _likelihood(ca, D, y_demeaned, n, model):
    R = model(D, ca)
    R_inv = la.inv(R)
    L = np.log(la.det(R)) + y_demeaned.dot(R_inv).dot(y_demeaned)
    return L


# Kernel reg of empirical data
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


# Empirical (distance, squared difference)
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


# Variogram Models
def model_factory(model_name):
    if model_name == 'exp':
        return gamma_exp
    else:
        # XXX TODO
        raise NotImplementedError


def gamma_exp(h, s):
    sig2, a = s
    g = sig2 * (np.exp(- h / a))
    return g

def exp_nug(h, s):
    sig2, a, nug = s
    g = 1 - (nug + (sig2 - nug) * (1 - np.exp(-h / a))) / sig2
    return g

def spherical(h, s):
    sigma2, a = s
    g = sigma2 * (1 - 1.5 * h / a + .5 * (h / a) ** 3)
    g[h > a] = 0
    return g

def gauss(h, s):
    sig2, a = s
    g = sig2 * (np.exp(-(h / a) ** 2))
    return g


if __name__ == "__main__":
    pass
