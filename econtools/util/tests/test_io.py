import tempfile
from os import remove, close, path
import shutil
import glob

import numpy as np
import pandas as pd

import nose
from nose.tools import assert_raises
from pandas.util.testing import assert_frame_equal

from econtools.util.io import _set_filepath, read, write, try_pickle

# TODO 'dec_load_or_build` (can this even be done?)
# TODO `load_or_build` (can this even be done?)


class test_set_filepath(object):

    def setup(self):
        self.args = (1990, 'tria')
        self.kwargs = {'base': 'house'}
        self.argspec = (
            ['year', 'model', 'base'],
            None,
            None,
            ['grid']
        )

    def test_args1(self):
        template = 'file_{}_{}'
        expected = 'file_1990_tria'
        path_args = [0, 1]
        result = _set_filepath(template, path_args, self.args, self.kwargs,
                               self.argspec)
        assert expected == result

    def test_args2(self):
        template = 'file_{}_{}'
        expected = 'file_house_1990'
        path_args = ['base', 0]
        result = _set_filepath(template, path_args, self.args, self.kwargs,
                               self.argspec)
        assert expected == result

    def test_noargs(self):
        expected = 'file'
        path_args = []
        result = _set_filepath(expected, path_args, self.args, self.kwargs,
                               self.argspec)
        assert expected == result

    def test_too_few_args(self):
        path_args = []
        assert_raises(ValueError, _set_filepath, 'file{}', path_args, self.args,
                      self.kwargs, self.argspec)

    def test_badarg_float(self):
        path_args = [1.0]
        assert_raises(ValueError, _set_filepath, '', path_args, self.args,
                      self.kwargs, self.argspec)


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
        self.aux_tempfile('hdf')
        write(self.df, self.tempfile)
        result = read(self.tempfile)
        assert_frame_equal(self.df, result)


class test_try_pickle(object):

    def setup(self):
        self.df = pd.DataFrame(np.arange(12).reshape(-1, 3),
                               columns=['a', 'b', 'c'])
        self.df.index.name = 'index'
        self.tempdir = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.tempdir)

    def test_already_p(self):
        temppath = path.join(self.tempdir, 'this_temp.p')
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
        expect_list = [fileroot.format(x) for x in ('p', 'dta')]
        assert set(result_list) == set(expect_list)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-v'])
