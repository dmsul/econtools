from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.api import reg
from econtools.metrics.tests.data.src_areg import (areg_std, areg_robust,
                                                   areg_cluster)


class AregCompare(RegCompare):

    def init(self):
        super(AregCompare, self).init(self)
        self.precision['ssr'] = 6

    def test_r2(self):
        pass

    def test_r2_a(self):
        pass

    def test_mss(self):
        pass


class TestAreg_std(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        fe_name = 'gear_ratio'
        cls.result = reg(autodata, y, x, fe_name=fe_name)
        cls.expected = areg_std


class TestAreg_hc1(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        fe_name = 'gear_ratio'
        cls.result = reg(autodata, y, x, fe_name=fe_name, vce_type='hc1')
        cls.expected = areg_robust


class TestAreg_cluster(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        fe_name = 'gear_ratio'
        cls.result = reg(autodata, y, x, fe_name=fe_name, cluster=fe_name)
        cls.expected = areg_cluster


if __name__ == '__main__':
    import pytest
    pytest.main()
