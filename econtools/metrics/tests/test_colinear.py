from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.core import _get_colinear_cols


class TestCheckColinear(RegCompare):

    def setup(self):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        self.df = autodata

    def test_one_colin_column(self):
        df = self.df.copy()
        df['neg_mpg'] = -1 * df['mpg']
        result = _get_colinear_cols(df[['price', 'mpg', 'neg_mpg']])
        expected = ['neg_mpg']
        assert result == expected


if __name__ == '__main__':
    import pytest
    pytest.main()
