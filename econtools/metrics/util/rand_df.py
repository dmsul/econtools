import pandas as pd
import numpy as np


def basic_samp(N=1000, K=10, cats=10):
    np.random.seed(1234567)

    x = np.random.normal(5., 10, (N, K))
    betas = np.arange(K)
    # Category fixed FX
    cat = np.random.randint(0, cats, N)
    cat_fx = np.random.normal(10., 15, cats)
    cat_fx_matrix = np.zeros(N)
    for i in range(N):
        cat_fx_matrix[i] = cat_fx[cat[i]]
    # outcome
    y = cat_fx_matrix + np.dot(x, betas) + np.random.normal(0, 30, N)
    colnames = ['y'] + ['x{}'.format(i) for i in range(K)] + ['cat']
    df = pd.DataFrame(np.column_stack([y, x, cat]), columns=colnames)
    return df
