from os import path

import pandas as pd

from econtools.metrics.api import reg, ivreg


class TestOLS_savemem(object):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        cls.result = reg(autodata, y, x, addcons=True, save_mem=True)

    def test_sample(self):
        assert not hasattr(self.result, 'sample')

    def test_resid(self):
        assert not hasattr(self.result, 'resid')

    def test_yhat(self):
        assert not hasattr(self.result, 'yhat')


class TestTsls_savemem(object):

    @classmethod
    def setup_class(cls):
        """Stata reg output from `sysuse auto; reg price mpg`"""
        test_path = path.split(path.relpath(__file__))[0]
        auto_path = path.join(test_path, 'data', 'auto.dta')
        autodata = pd.read_stata(auto_path)
        y = 'price'
        x = ['mpg', 'length']
        z = ['trunk', 'weight']
        w = []
        cls.result = ivreg(autodata, y, x, z, w, addcons=True, save_mem=True)

    def test_sample(self):
        assert not hasattr(self.result, 'sample')

    def test_resid(self):
        assert not hasattr(self.result, 'resid')

    def test_yhat(self):
        assert not hasattr(self.result, 'yhat')


if __name__ == '__main__':
    import pytest
    pytest.main()
