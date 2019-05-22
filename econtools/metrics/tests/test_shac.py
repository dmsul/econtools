"""
Note: Doing SHAC with a trivially small bandwidth is numerically identical to
clustering. So the clustering results can be used as the "correct" value of the
SHAC with trivially small bandwidth (so the only obs that fall in unit i's
bandwidth are unit i's). However, the degress of freedom corrections may be
different, so this must be adjusted for.
"""
import pandas as pd
import numpy as np

from numpy.testing import (assert_array_almost_equal,)

from econtools.util.frametools import group_id

from econtools.metrics.api import reg


class SHACRegCompare(object):

    def setup(cls):
        N = 100
        df = pd.DataFrame()
        df['x1'] = 5*np.random.randn(N) + 12
        df['lat'] = np.random.rand(N)
        df['lon'] = np.random.rand(N)
        p = [(.25, .25), (.5, .5), (.75, .75)]
        d = np.zeros((N, len(p)))
        for i in range(len(p)):
            d[:, i] = np.sqrt(
                (df['lon'] - p[i][0])**2 +
                (df['lat'] - p[i][1])**2
            )
        point_effect = (np.random.rand(len(p)) - .5)*10
        full_point_effect = d.dot(point_effect)

        df['y'] = 12 + 4*df['x1'] + full_point_effect + np.random.randn(N)*20

        df['lat_grid'] = np.around(df['lat']*10)
        df['lon_grid'] = np.around(df['lon']*10)

        cls.y = 'y'
        cls.x = ['x1']
        cls.N = N
        cls.cluster = 'hgrid'
        cls.lat = 'lat_grid'
        cls.lon = 'lon_grid'
        cls.spatial_hac = dict(x=cls.lon, y=cls.lat, kern='unif', band=0.01)
        df = group_id(df, cols=[cls.lon, cls.lat], name=cls.cluster,
                      merge=True)
        cls.df = df


class TestSHAC_reg(SHACRegCompare):

    def test_reg_unif(self):
        std = reg(self.df, self.y, self.x, cluster=self.cluster, addcons=True)
        shac = reg(self.df, self.y, self.x, shac=self.spatial_hac,
                   addcons=True)
        expected = std.vce / std._vce_correct
        result = shac.vce / shac._vce_correct
        assert_array_almost_equal(expected, result)

    def test_reg_tria(self):
        std = reg(self.df, self.y, self.x, cluster=self.cluster, addcons=True)
        self.spatial_hac['kern'] = 'tria'
        shac = reg(self.df, self.y, self.x, shac=self.spatial_hac,
                   addcons=True)
        expected = std.vce / std._vce_correct
        result = shac.vce / shac._vce_correct
        assert_array_almost_equal(expected, result)


if __name__ == '__main__':
    import pytest
    pytest.main()
