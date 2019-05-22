import numpy as np

from numpy.testing import (assert_array_equal, assert_array_almost_equal,)

class RegCompare(object):

    def init(self):
        self.precision = {
            'coeff': 6,
            'se': 6,
            't': 6,
            'p>t': 6,
            'CI_low': 4,
            'CI_high': 5,
            'ssr': 7,
            'sst': 6,
            'mss': 6,
            'r2': 10,
            'r2_a': 10,
            'F': 10,
            'pF': 10,
            'vce': 10,
            'N': 10,
        }

    def summ_assert(self, stat):
        result = self.result.summary[stat].values
        expected = self.expected.summary[stat].values
        digits = self.precision[stat]
        self._assert_core(result, expected, digits)

    def stat_assert(self, stat):
        if stat == 'vce':
            raw_res = getattr(self.result, stat).values
            raw_exp = getattr(self.expected, stat).values
            result = (raw_res - raw_exp)/raw_exp
            expected = np.zeros(result.shape)
        else:
            result = getattr(self.result, stat)
            expected = getattr(self.expected, stat)
        digits = self.precision[stat]
        self._assert_core(result, expected, digits)

    def _assert_core(self, result, expected, digits):
        if digits <= 0:
            scale = 10**(-digits)
            assert_array_equal(np.around(result/scale)*scale,
                               np.around(expected/scale)*scale)
        else:
            assert_array_almost_equal(result, expected, decimal=digits)

    def test_coeff(self):
        stat = 'coeff'
        self.summ_assert(stat)

    def test_se(self):
        stat = 'se'
        self.summ_assert(stat)

    def test_t(self):
        stat = 't'
        self.summ_assert(stat)

    def test_pt(self):
        stat = 'p>t'
        self.summ_assert(stat)

    def test_cilow(self):
        stat = 'CI_low'
        self.summ_assert(stat)

    def test_cihigh(self):
        stat = 'CI_high'
        self.summ_assert(stat)

    def test_N(self):
        stat = 'N'
        self.stat_assert(stat)

    def test_ssr(self):
        stat = 'ssr'
        result = self.result.ssr
        expected = self.expected.rss
        assert_array_almost_equal(result, expected,
                                  decimal=self.precision[stat])

    def test_mss(self):
        stat = 'mss'
        result = self.result.sst - self.result.ssr,
        expected = self.expected.__dict__[stat]
        assert_array_almost_equal(result, expected,
                                  decimal=self.precision[stat])

    def test_r2(self):
        stat = 'r2'
        self.stat_assert(stat)

    def test_r2_a(self):
        stat = 'r2_a'
        self.stat_assert(stat)

    def test_Fstat(self):
        stat = 'F'
        self.stat_assert(stat)

    def test_pF(self):
        stat = 'pF'
        self.stat_assert(stat)

    def test_vce(self):
        stat = 'vce'
        self.stat_assert(stat)
