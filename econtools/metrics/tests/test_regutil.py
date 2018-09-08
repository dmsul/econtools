import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal

from econtools.metrics.regutil import winsorize


class TestWinsorize(object):

    @classmethod
    def setup_class(cls):
        df = pd.DataFrame(np.column_stack((
            np.arange(21),
            np.array([50, 2.4] + [5]*19))),
            columns=['x', 'y'])
        cls.df = df

    def test_onevar(self):
        expected = self.df.iloc[2:, :]
        result = winsorize(self.df, 'y')
        assert_frame_equal(expected, result)

    def test_twovar(self):
        expected = self.df.iloc[2:-1, :]
        result = winsorize(self.df, ('x', 'y'))
        assert_frame_equal(expected, result)

    def test_onesided(self):
        expected = self.df.iloc[:-1, :]
        result = winsorize(self.df, 'x', p=(0, .95))
        print(expected)
        print(result)
        assert_frame_equal(expected, result)

    def test_separate_quantiles(self):
        df = self.df.copy()
        df.iloc[10, 0] = 99
        p = [(0, .99), (.01, 1)]
        kept_index = [x for x in df.index if x not in [1, 10]]
        expected = df.iloc[kept_index, :]
        result = winsorize(df, ('x', 'y'), p=p)
        assert_frame_equal(expected, result)


if __name__ == '__main__':
    import pytest
    pytest.main()
