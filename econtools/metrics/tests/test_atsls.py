from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.core import ivreg
from data.src_atsls import atsls_std, atsls_robust, atsls_cluster
from data.src_atsls_nosing import (atsls_nosing_std, atsls_nosing_robust,
                                   atsls_nosing_cluster)


class AtslsCompare(RegCompare):

    def __init__(self):
        super(AtslsCompare, self).__init__()
        self.precision['coeff'] = 4
        self.precision['vce'] = 8
        self.precision['se'] = 3
        self.precision['CI_low'] = 2
        self.precision['CI_high'] = 1

    def test_r2(self):
        pass

    def test_r2_a(self):
        pass

    def test_mss(self):
        pass

    def test_ssr(self):
        # Stata SSR for IV is weird, skip
        pass


class TestAtsls_std(AtslsCompare):

    def __init__(self):
        super(TestAtsls_std, self).__init__()

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        a = 'gear_ratio'
        nosingles = False
        cls.result = ivreg(autodata, y, x_end, z, x_exog, a_name=a,
                           nosingles=nosingles)
        cls.expected = atsls_std


class TestAtsls_robust(AtslsCompare):

    def __init__(self):
        super(TestAtsls_robust, self).__init__()
        # The SE's in this regression are huge, Tons of floating point error in
        # VCE, gets passed along to everything
        self.precision['vce'] = 7
        self.precision['F'] = 6
        self.precision['pF'] = 6
        self.precision['se'] = 1
        self.precision['CI_low'] = 1

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        a = 'gear_ratio'
        nosingles = False
        cls.result = ivreg(autodata, y, x_end, z, x_exog, a_name=a,
                           vce_type='robust', nosingles=nosingles)
        cls.expected = atsls_robust


class TestAtsls_cluster(AtslsCompare):

    def __init__(self):
        super(TestAtsls_cluster, self).__init__()

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        a = 'gear_ratio'
        nosingles = False
        cls.result = ivreg(autodata, y, x_end, z, x_exog, a_name=a, cluster=a,
                           nosingles=nosingles)
        cls.expected = atsls_cluster


class TestAtsls_nosing_std(AtslsCompare):

    def __init__(self):
        super(TestAtsls_nosing_std, self).__init__()

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        a = 'gear_ratio'
        nosingles = True
        cls.result = ivreg(autodata, y, x_end, z, x_exog, a_name=a,
                           nosingles=nosingles)
        cls.expected = atsls_nosing_std


class TestAtsls_nosing_robust(AtslsCompare):

    def __init__(self):
        super(TestAtsls_nosing_robust, self).__init__()
        # The SE's in this regression are huge, Tons of floating point error in
        # VCE, gets passed along to everything
        self.precision['vce'] = 7
        self.precision['F'] = 6
        self.precision['pF'] = 6
        self.precision['se'] = 1
        self.precision['CI_low'] = 1

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        a = 'gear_ratio'
        nosingles = True
        cls.result = ivreg(autodata, y, x_end, z, x_exog, a_name=a,
                           vce_type='robust', nosingles=nosingles)
        cls.expected = atsls_nosing_robust


class TestAtsls_nosing_cluster(AtslsCompare):

    def __init__(self):
        super(TestAtsls_nosing_cluster, self).__init__()

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        a = 'gear_ratio'
        nosingles = True
        cls.result = ivreg(autodata, y, x_end, z, x_exog, a_name=a, cluster=a,
                           nosingles=nosingles)
        cls.expected = atsls_nosing_cluster


if __name__ == '__main__':
    import sys
    from nose import runmodule
    argv = [__file__, '-vs'] + sys.argv[1:]
    runmodule(argv=argv, exit=False)
