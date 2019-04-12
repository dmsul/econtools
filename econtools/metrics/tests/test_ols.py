from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.api import reg
from econtools.metrics.tests.data.src_ols import (ols_std, ols_robust, ols_hc2,
                                                  ols_hc3, ols_cluster)


class TestOLS_std(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['vce'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, addcons=True)
        cls.expected = ols_std


class TestOLS_std_y_list(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['vce'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = ['price']
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, addcons=True)
        cls.expected = ols_std


class TestOLS_hc1(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, vce_type='hc1', addcons=True)
        cls.expected = ols_robust


class TestOLS_hc2(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, vce_type='hc2', addcons=True)
        cls.expected = ols_hc2


class TestOLS_hc3(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, vce_type='hc3', addcons=True)
        cls.expected = ols_hc3


class TestOLS_cluster(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, cluster='gear_ratio', addcons=True)
        cls.expected = ols_cluster


if __name__ == '__main__':
    import pytest
    pytest.main()
