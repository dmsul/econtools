from __future__ import division

from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt

from econtools.metrics.locallinear import llr


def prelim_gamma(X, z, maxd=None, scat=False, **npregargs):
    # Calc distances
    dist = pdist(X)
    # Calc squared diff of z
    N = len(X)
    sqdiff = np.zeros(len(dist))
    for i in range(N):
        z_i = z[i]
        for j in range(i + 1, N):
            sqdiff[get_flat_matrix_idx(i, j, N)] = (z_i - z[j]) ** 2
    if maxd is not None:
        sqdiff = sqdiff[dist < maxd]
        dist = dist[dist < maxd]
    # plot w/ kernel reg
    xG, est_stats = llr(sqdiff, dist, **npregargs)
    fig, ax = plt.subplots()
    if scat:
        ax.scatter(dist, sqdiff)
    ax.plot(xG[:, 0], xG[:, 1], '-og')
    plt.show()
    return est_stats


def get_flat_matrix_idx(i, j, n):
    return n*i - i*(i + 1) // 2 + j - i - 1


if __name__ == "__main__":
    pass
