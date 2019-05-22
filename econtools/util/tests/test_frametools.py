import pytest

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal

from econtools.util.frametools import stata_merge, group_id, winsorize


class TestStata_merge(object):

    def setup(self):
        """Setup before each test method"""
        self.df1 = pd.DataFrame(np.array([[0, 1, 2], [1, 1, 1]]).T,
                                columns=['a', 'b'],
                                dtype=np.float64)
        self.df1_match = pd.DataFrame(np.array([[0, 1, 2], [11, 11, 11]]).T,
                                      columns=['a', 'c'],
                                      dtype=np.float64)
        self.df1_part = pd.DataFrame(np.array([[0, 1, 4], [9, 10, 11]]).T,
                                     columns=['a', 'c'],
                                     dtype=np.float64)

    def test_assert3(self):
        result = stata_merge(self.df1, self.df1_match, on='a', how='left',
                             assertval=3)
        expected = pd.DataFrame(
            np.array([[0, 1, 2], [1, 1, 1], [11, 11, 11]]).T,
            columns=['a', 'b', 'c'],
            dtype=np.float64)
        assert_frame_equal(result, expected)

    def test_part_left(self):
        result = stata_merge(self.df1, self.df1_part, on='a', how='left')
        expected = pd.DataFrame(
            np.array([[0, 1, 2], [1, 1, 1], [9, 10, np.nan], [3, 3, 1]]).T,
            columns=['a', 'b', 'c', '_m'],
            dtype=np.float64)
        print(result)
        print(expected)
        assert_frame_equal(result, expected)

    def test_part_outer(self):
        result = stata_merge(self.df1, self.df1_part, on='a', how='outer')
        expected = pd.DataFrame(
            np.array([[0, 1, 2, 4], [1, 1, 1, np.nan], [9, 10, np.nan, 11],
                     [3, 3, 1, 2]]).T,
            columns=['a', 'b', 'c', '_m'],
            dtype=np.float64)
        print(result)
        print(expected)
        assert_frame_equal(result, expected)

    def test_fail3(self):
        with pytest.raises(AssertionError):
            stata_merge(self.df1, self.df1_part, on='a', how='left',
                        assertval=3)

    def test_autonaming(self):
        left = self.df1.rename(columns={'a': 'tmpa'})
        right = self.df1_match.rename(columns={'a': 'tmpb'})
        result = stata_merge(left, right, left_on='tmpa', right_on='tmpb',
                             how='left', assertval=3)
        expected = pd.DataFrame(
            np.array([[0, 1, 2], [1, 1, 1], [0, 1, 2], [11, 11, 11]]).T,
            columns=['tmpa', 'b', 'tmpb', 'c'],
            dtype=np.float64)
        assert_frame_equal(result, expected)


class TestGroupid(object):

    @classmethod
    def setup_class(self):
        self.df = pd.DataFrame(np.array([
            [1, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [1, 2]]),
            columns=['x', 'y'],
            index=['a', 'b', 'c', 'd', 'e'])

        tmpdf = self.df.copy()
        tmpdf['group_id'] = [0, 0, 1, 2, 1]
        self.df_with_id = tmpdf[['group_id', 'x', 'y']]

        tmpdf = self.df_with_id.drop_duplicates().sort_values('group_id')
        tmpdf = tmpdf.reset_index(drop=True)
        self.dfs_id_xwalk = tmpdf

    def test_group_id(self):
        expected = self.dfs_id_xwalk
        result = group_id(self.df)
        assert_frame_equal(expected, result)

    def test_subset_of_cols(self):
        expected = self.dfs_id_xwalk
        df = self.df.copy()
        df['z'] = np.arange(df.shape[0])
        result = group_id(df, cols=['x', 'y'])
        assert_frame_equal(expected, result)

    def test_bad_idname_error(self):
        with pytest.raises(ValueError):
            group_id(self.df, name='x')

    def test_automerge(self):
        expected = self.df_with_id[['x', 'y', 'group_id']]
        result = group_id(self.df, merge=True)
        print('\n')
        print(expected)
        print(result)
        assert_frame_equal(expected, result)


class TestWinsorize(object):

    @classmethod
    def setup_class(cls):
        df = pd.DataFrame(np.column_stack((
            np.arange(21),
            np.array([50, 2.4] + [5]*19))),
            columns=['x', 'y'])
        cls.df = df

    def test_onevar(self):
        expected = self.df.iloc[2:, :]
        result = winsorize(self.df, 'y')
        assert_frame_equal(expected, result)

    def test_twovar(self):
        expected = self.df.iloc[2:-1, :]
        result = winsorize(self.df, ('x', 'y'))
        assert_frame_equal(expected, result)

    def test_onesided(self):
        expected = self.df.iloc[:-1, :]
        result = winsorize(self.df, 'x', p=(0, .95))
        print(expected)
        print(result)
        assert_frame_equal(expected, result)

    def test_separate_quantiles(self):
        df = self.df.copy()
        df.iloc[10, 0] = 99
        p = [(0, .99), (.01, 1)]
        kept_index = [x for x in df.index if x not in [1, 10]]
        expected = df.iloc[kept_index, :]
        result = winsorize(df, ('x', 'y'), p=p)
        assert_frame_equal(expected, result)


if __name__ == '__main__':
    pytest.main()
