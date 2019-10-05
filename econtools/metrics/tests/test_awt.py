from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.api import reg
from econtools.metrics.tests.data.src_awt import (awt_std, awt_robust, awt_hc2,
                                                  awt_hc3, awt_cluster)


class TestAwt_std(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['vce'] = 6
        cls.precision['ssr'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, addcons=True, awt_name='mpg')
        cls.expected = awt_std


class TestAwt_hc1(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['ssr'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, vce_type='hc1', addcons=True,
                         awt_name='mpg')
        cls.expected = awt_robust


class TestAwt_hc2(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['ssr'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, vce_type='hc2', addcons=True,
                         awt_name='mpg')
        cls.expected = awt_hc2


class TestAwt_hc3(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['ssr'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, vce_type='hc3', addcons=True,
                         awt_name='mpg')
        cls.expected = awt_hc3


class TestAwt_cluster(RegCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        cls.precision['ssr'] = 6
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, cluster='gear_ratio', addcons=True,
                         awt_name='mpg')
        cls.expected = awt_cluster


if __name__ == '__main__':
    import pytest
    pytest.main()
