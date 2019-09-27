from os import path

import pandas as pd
import numpy as np
import pytest

from econtools.metrics.api import reg
from econtools.metrics.core import _get_colinear_cols


class TestCheckColinear(object):

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

    def test_two_colin_column(self):
        df = self.df.copy()
        df['neg_mpg'] = -1 * df['mpg']
        df['mpg2'] = 2 * df['mpg']
        result = _get_colinear_cols(df[['price', 'mpg', 'neg_mpg', 'mpg2']])
        expected = ['neg_mpg', 'mpg2']
        assert result == expected

    def test_two_colin_column_reorder(self):
        df = self.df.copy()
        df['neg_mpg'] = -1 * df['mpg']
        df['mpg2'] = 2 * df['mpg']
        result = _get_colinear_cols(df[['mpg2', 'mpg', 'price', 'neg_mpg']])
        expected = ['mpg', 'neg_mpg']
        assert result == expected


class TestColinearReg(object):

    def test_colinear_reg(self):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        autodata['mpg2'] = autodata['mpg'] * 2
        y = 'price'
        x = ['mpg', 'length', 'mpg2']
        # Check without check (Singular matrix error)
        with pytest.raises(np.linalg.linalg.LinAlgError):
            reg(autodata, y, x, addcons=True)
        # Check with check (my ValueError, with message)
        with pytest.raises(ValueError) as e:
            reg(autodata, y, x, addcons=True, check_colinear=True)
        assert 'Colinear variables: \nmpg2' == str(e.value)


if __name__ == '__main__':
    pytest.main()
