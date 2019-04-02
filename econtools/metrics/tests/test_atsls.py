from os import path

import pandas as pd

from econtools.metrics.util.testing import RegCompare
from econtools.metrics.api import ivreg
from econtools.metrics.tests.data.src_atsls import atsls_std


class AtslsCompare(RegCompare):

    def init(self):
        super(AtslsCompare, self).init(self)
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

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        cls.init(cls)
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x_end = ['mpg', 'length']
        z = ['trunk', 'weight']
        x_exog = []
        fe = 'gear_ratio'
        cls.result = ivreg(autodata, y, x_end, z, x_exog, fe_name=fe)
        cls.expected = atsls_std


if __name__ == '__main__':
    import pytest
    pytest.main()
