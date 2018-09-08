import tempfile
from os import remove, close, path
import shutil
import glob

import numpy as np
import pandas as pd

import pytest

from pandas.util.testing import assert_frame_equal

from econtools.util.io import (_set_filepath_old, read, write, try_pickle,
                               load_or_build)
# TODO 'dec_load_or_build` (can this even be done?)


def builder(year, model, base='house'):
    pass


class Test_load_or_build(object):

    def setup(self):
        self.df = pd.DataFrame(np.arange(12).reshape(-1, 3),
                               columns=['a', 'b', 'c'])
        self.df.index.name = 'index'
        self.tmppath_root = path.join(
            path.split(path.relpath(__file__))[0],
            'tmp'
        )

    def teardown(self):
        tmppath = path.join(
            path.split(path.relpath(__file__))[0],
            'tmp*.pkl'
        )
        paths = glob.glob(tmppath)
        for fpath in paths:
            remove(fpath)

    def test_basic(self):

        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '.pkl'

        @load_or_build(tmppath)
        def lob_builder():
            df = expected.copy()

            return df

        lob_builder()
        result = pd.read_pickle(tmppath)
        assert_frame_equal(expected, result)

    def test_arg(self):

        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '_{year}.pkl'

        @load_or_build(tmppath)
        def lob_builder(year):
            df = expected.copy()
            df = df.rename(columns={'a': year})

            return df

        year = 1999
        lob_builder(year)
        expected = expected.rename(columns={'a': year})
        result = pd.read_pickle(tmppath.format(year=year))
        assert_frame_equal(expected, result)

    def test_kwarg_default(self):

        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '_{year}.pkl'

        @load_or_build(tmppath)
        def lob_builder(year=2001):
            df = expected.copy()
            df = df.rename(columns={'a': year})

            return df

        lob_builder()
        expected = expected.rename(columns={'a': 2001})
        result = pd.read_pickle(tmppath.format(year=2001))
        assert_frame_equal(expected, result)

    def test_kwarg_passed(self):

        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '_{year}.pkl'

        @load_or_build(tmppath)
        def lob_builder(year=2001):
            df = expected.copy()
            df = df.rename(columns={'a': year})

            return df

        lob_builder(year=2002)
        expected = expected.rename(columns={'a': 2002})
        result = pd.read_pickle(tmppath.format(year=2002))
        assert_frame_equal(expected, result)

    def test_arg_w_kwarg_default(self):

        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '_{base}_{year}.pkl'

        @load_or_build(tmppath)
        def lob_builder(base, year=2001):
            df = expected.copy()
            df = df.rename(columns={'a': year, 'b': base})

            return df

        lob_builder('house')
        expected = expected.rename(columns={'a': 2001, 'b': 'house'})
        result = pd.read_pickle(tmppath.format(base='house', year=2001))
        assert_frame_equal(expected, result)

    def test_arg_w_kwarg_passed(self):

        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '_{base}_{year}.pkl'

        @load_or_build(tmppath)
        def lob_builder(base, year=2001):
            df = expected.copy()
            df = df.rename(columns={'a': year, 'b': base})

            return df

        lob_builder('house', year=1976)
        expected = expected.rename(columns={'a': 1976, 'b': 'house'})
        result = pd.read_pickle(tmppath.format(base='house', year='1976'))
        assert_frame_equal(expected, result)

    def test_load_direct(self):
        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '.pkl'

        @load_or_build(tmppath)
        def lob_builder():
            df = expected.copy()

            return df

        lob_builder(_load=False)
        assert not path.isfile(tmppath)

    def test_rebuild(self):
        old = pd.DataFrame(np.arange(12).reshape(-1, 3),
                           columns=['a', 'b', 'c'])
        old.index.name = 'index'
        tmp_path = self.tmppath_root + '_rebuild.pkl'

        old.to_pickle(tmp_path)
        new = old.copy()
        new = new.rename(columns={'a': 'DDD'})

        @load_or_build(tmp_path)
        def lob_builder():
            df = new.copy()

            return df

        # Check that it reads the old one
        from_disk = lob_builder()
        assert_frame_equal(from_disk, old)
        # Check that it returns the new one w/ `_rebuild=True`
        rebuilt = lob_builder(_rebuild=True)
        assert_frame_equal(rebuilt, new)
        # Check that the new one is on disk after `_rebuild=True`
        rebuilt_from_disk = pd.read_pickle(tmp_path)
        assert_frame_equal(rebuilt_from_disk, new)

    def test_load_direct_and_rebuild(self):
        # If both _load=False and _rebuild=True as switched, _rebuild=True
        # should be ignored. Add warning?
        expected = pd.DataFrame(np.arange(12).reshape(-1, 3),
                                columns=['a', 'b', 'c'])
        expected.index.name = 'index'
        tmppath = self.tmppath_root + '.pkl'

        @load_or_build(tmppath)
        def lob_builder():
            df = expected.copy()

            return df

        lob_builder(_load=False, _rebuild=True)
        assert not path.isfile(tmppath)


class Test_set_filepath_old(object):

    def setup(self):
        self.args = (1990, 'tria')
        self.kwargs = {'base': 'house'}

    def test_args1(self):
        template = 'file_{}_{}'
        expected = 'file_1990_tria'
        path_args = [0, 1]
        result = _set_filepath_old(template, path_args, self.args, self.kwargs,
                                   builder)
        assert expected == result

    def test_args2(self):
        template = 'file_{}_{}'
        expected = 'file_house_1990'
        path_args = ['base', 0]
        result = _set_filepath_old(template, path_args, self.args, self.kwargs,
                                   builder)
        assert expected == result

    def test_noargs(self):
        expected = 'file'
        path_args = []
        result = _set_filepath_old(expected, path_args, self.args, self.kwargs,
                                   builder)
        assert expected == result

    def test_too_few_args(self):
        path_args = []
        with pytest.raises(ValueError):
            _set_filepath_old('file{}', path_args, self.args, self.kwargs,
                              builder)

    def test_badarg_float(self):
        path_args = [1.0]
        with pytest.raises(ValueError):
            _set_filepath_old('', path_args, self.args, self.kwargs, builder)


class test_readwrite(object):

    def setup(self):
        self.df = pd.DataFrame(np.arange(12).reshape(-1, 3),
                               columns=['a', 'b', 'c'])
        self.df.index.name = 'index'

    def aux_tempfile(self, suffix):
        self.fd, self.tempfile = tempfile.mkstemp(suffix='.{}'.format(suffix))

    def teardown(self):
        close(self.fd)          # Close the file descriptor
        remove(self.tempfile)

    def test_csv(self):
        # NOTE: pass an int32 to csv, it comes back int64, don't know why
        self.aux_tempfile('csv')
        csv_df = self.df.astype(np.int64)
        write(csv_df, self.tempfile)
        result = read(self.tempfile).set_index('index')
        assert_frame_equal(csv_df, result)

    def test_dta(self):
        self.aux_tempfile('dta')
        write(self.df, self.tempfile)
        result = read(self.tempfile).set_index('index')
        assert_frame_equal(self.df, result)

    def test_p(self):
        self.aux_tempfile('p')
        write(self.df, self.tempfile)
        result = read(self.tempfile)
        assert_frame_equal(self.df, result)

    def test_hdf(self):
        self.aux_tempfile('hdf5')
        write(self.df, self.tempfile)
        result = read(self.tempfile)
        assert_frame_equal(self.df, result)


class Test_try_pickle(object):

    def setup(self):
        self.df = pd.DataFrame(np.arange(12).reshape(-1, 3),
                               columns=['a', 'b', 'c'])
        self.df.index.name = 'index'
        self.tempdir = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.tempdir)

    def test_already_p(self):
        temppath = path.join(self.tempdir, 'this_temp.pkl')
        self.df.to_pickle(temppath)
        result = try_pickle(temppath)
        assert_frame_equal(self.df, result)
        file_list = glob.glob(path.join(self.tempdir, '*'))
        assert len(file_list) == 1

    def test_needs_p(self):
        fileroot = path.join(self.tempdir, 'this_temp.{}')
        temppath = fileroot.format('dta')
        self.df.to_stata(temppath)
        try_pickle(temppath)
        result_list = glob.glob(path.join(self.tempdir, '*'))
        expect_list = [fileroot.format(x) for x in ('pkl', 'dta')]
        assert set(result_list) == set(expect_list)


if __name__ == '__main__':
    pass
