import pandas as pd
import numpy as np
import pytest

from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal

from econtools.util.gentools import (int2base, base2int, force_df, force_list,
                                     force_iterable)


class Test_BaseConvert(object):

    def test_2b16_1(self):
        x = 17
        expected = '11'
        result = int2base(x, 16)
        assert expected == result

    def test_2b36_1(self):
        x = 37
        expected = '11'
        result = int2base(x, 36)
        assert expected == result

    def test_2b62_1(self):
        x = 63
        expected = '11'
        result = int2base(x, 62)
        assert expected == result

    def test_b16(self):
        base = 16
        x = 1022
        to = int2base(x, base)
        back = base2int(to, base)
        assert x == back

    def test_b36(self):
        base = 36
        x = 1022
        to = int2base(x, base)
        back = base2int(to, base)
        assert x == back

    def test_b62(self):
        base = 36
        x = 1022
        to = int2base(x, base)
        back = base2int(to, base)
        assert x == back

    def test_invalid_char_b16(self):
        with pytest.raises(ValueError):
            base2int('123Z', 16)

    def test_invalid_char_b32(self):
        with pytest.raises(ValueError):
            base2int('123a', 32)

    def test_invalid_char_b62(self):
        with pytest.raises(ValueError):
            base2int('123!', 62)


class Test_force_df(object):

    def setup(self):
        s = pd.Series([1, 2, 3], name='wut')
        self.s = s
        self.sdf = pd.DataFrame(s)
        df = pd.DataFrame([s, s]).T
        df.columns = ['dingle', 'dangle']
        self.df = df

    def test_simple(self):
        expected = pd.DataFrame(self.s)
        result = force_df(self.s)
        assert_frame_equal(expected, result)

    def test_passthrough(self):
        df = self.df
        assert_frame_equal(df, force_df(df))

    def test_multiidx(self):
        idx = pd.MultiIndex.from_tuples([('a', 1), ('b', 7), ('w', 99)],
                                        names=['dingle', 'dangle'])
        expected = self.sdf
        expected.index = idx
        s = self.s
        s.index = idx
        result = force_df(s)
        assert_frame_equal(expected, result)

    def test_list(self):
        with pytest.raises(ValueError):
            force_df([1, 2, 3])

    def test_array(self):
        with pytest.raises(ValueError):
            force_df(np.arange(3))


class Test_force_list(object):

    def test_list(self):
        expected = [1, 2, 3]
        result = force_list(expected)
        assert expected == result

    def test_int(self):
        an_int = 10
        expected = [an_int]
        result = force_list(an_int)
        assert expected == result

    def test_tup(self):
        a_tuple = (1, 2, 3)
        expected = list(a_tuple)
        result = force_list(a_tuple)
        assert expected == result

    def test_array(self):
        an_array = np.arange(3)
        expected = an_array.tolist()
        result = force_list(an_array)
        assert expected == result

    def test_series(self):
        a_series = pd.Series(np.arange(3))
        expected = a_series.tolist()
        result = force_list(a_series)
        assert expected == result

    def test_string(self):
        a_string = 'abcd'
        expected = [a_string]
        result = force_list(a_string)
        assert expected == result


class Test_force_iterable(object):

    def test_list(self):
        expected = [1, 2, 3]
        result = force_iterable(expected)
        assert expected == result

    def test_int(self):
        an_int = 10
        expected = (an_int,)
        result = force_iterable(an_int)
        assert expected == result

    def test_tup(self):
        expected = (1, 2, 3)
        result = force_iterable(expected)
        assert expected == result

    def test_array(self):
        expected = np.arange(3)
        result = force_iterable(expected)
        assert_array_equal(expected, result)

    def test_string(self):
        a_string = 'abcd'
        expected = (a_string,)
        result = force_iterable(a_string)
        assert expected == result


if __name__ == '__main__':
    pytest.main()
