from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.core import reg
from data.src_areg import areg_std, areg_robust, areg_cluster


class AregCompare(RegCompare):

    def __init__(self):
        super(AregCompare, self).__init__()
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
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        a_name = 'gear_ratio'
        cls.result = reg(autodata, y, x, a_name=a_name)
        cls.expected = areg_std


class TestAreg_hc1(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        a_name = 'gear_ratio'
        cls.result = reg(autodata, y, x, a_name=a_name, vce_type='hc1')
        cls.expected = areg_robust


class TestAreg_cluster(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        a_name = 'gear_ratio'
        cls.result = reg(autodata, y, x, a_name=a_name, cluster=a_name)
        cls.expected = areg_cluster


if __name__ == '__main__':
    import sys
    from nose import runmodule
    argv = [__file__, '-vs'] + sys.argv[1:]
    runmodule(argv=argv, exit=False)
