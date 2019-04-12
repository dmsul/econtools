import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal

from econtools.metrics.regutil import add_cons


class TestCheckCons(object):

    def setup(self):
        df = pd.DataFrame(np.arange(12).reshape(-1, 3),
                          columns=['a', 'b', 'c'])
        self.needs = df.copy()
        self.has = df.copy()
        self.has['c'] = np.ones(self.has.shape[0])
        self.needs_after = self.needs.copy()
        self.needs_after['_cons'] = np.ones(self.needs_after.shape[0])

    def test_addcons(self):
        result = add_cons(self.needs)
        expected = self.needs_after
        assert_frame_equal(expected, result)

    def test_add_nochange_orig(self):
        pass_to_func = self.needs.copy()
        result = add_cons(pass_to_func)
        assert_frame_equal(self.needs, pass_to_func)


if __name__ == '__main__':
    import pytest
    pytest.main()
