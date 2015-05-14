from os import path

import pandas as pd

from metrics.util.testing import RegCompare
from metrics.othregs import areg
from data.src_areg import areg_std, areg_robust, areg_cluster
from data.src_areg_nosing import (areg_nosing_std, areg_nosing_robust,
                                  areg_nosing_cluster)


class AregCompare(RegCompare):

    def __init__(self):
        super(AregCompare, self).__init__()
        self.precision['ssr'] = 6


class TestAreg_std(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        avar = 'gear_ratio'
        nosingles = False
        cls.result = areg(autodata, y, x, avar=avar, nosingles=nosingles)
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
        avar = 'gear_ratio'
        nosingles = False
        cls.result = areg(autodata, y, x, avar=avar, vce_type='hc1',
                          nosingles=nosingles)
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
        avar = 'gear_ratio'
        nosingles = False
        cls.result = areg(autodata, y, x, avar=avar, cluster=avar,
                          nosingles=nosingles)
        cls.expected = areg_cluster


class TestAreg_nosing_std(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        avar = 'gear_ratio'
        nosingles = True
        cls.result = areg(autodata, y, x, avar=avar, nosingles=nosingles)
        cls.expected = areg_nosing_std


class TestAreg_nosing_hc1(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        avar = 'gear_ratio'
        nosingles = True
        cls.result = areg(autodata, y, x, avar=avar, vce_type='hc1',
                          nosingles=nosingles)
        cls.expected = areg_nosing_robust


class TestAreg_nosing_cluster(AregCompare):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        avar = 'gear_ratio'
        nosingles = True
        cls.result = areg(autodata, y, x, avar=avar, cluster=avar,
                          nosingles=nosingles)
        cls.expected = areg_nosing_cluster


if __name__ == '__main__':
    import sys
    from nose import runmodule
    argv = [__file__, '-vs'] + sys.argv[1:]
    runmodule(argv=argv, exit=False)
