from os.path import isfile, splitext
from functools import wraps
import argparse

import pandas as pd

from .gentools import force_df


# TODO: Drop functional `load_or_build`, replace with decorator (once the usage
#   in projects is updated and tested).
def dec_load_or_build(filepath, copydta=False):
    """
    Loads `filepath` as a DataFrame if it exists, otherwise builds the data and
    saves it to `filepath`.

    Decorator Args
    ---------------
    `filepath`, str: path to DataFrame

    Decorator Kwargs
    ---------------
    `copydta`, bool: if true, save a copy of the data in Stata DTA format if
        `filepath` is not already a DTA file.

    Build Function Kwargs
    ---------------------
    This are additional kwargs that can be passed to the wrapped function that
        affect the behavior of `load_or_build`.

    `_rebuild`, bool (False): Build the DataFrame and save it to `filepath` even
        if `filepath` already exists.
    `_load`, bool (True): Try loading the data before building it. Like
        `_rebuild`, but no copy of the data is written to disk.
    """
    def actualDecorator(builder):
        @wraps(builder)
        def wrapper(*args, **kwargs):
            load = kwargs.pop('_load', True)
            rebuild = kwargs.pop('_rebuild', False)
            if load and isfile(filepath) and not rebuild:
                # If it's on disk and there are no objections, read it
                df = read(filepath)
            elif not load:
                # If `load = False` is passed, just return the built `df`
                df = builder(*args, **kwargs)
            else:
                # If it's just not on disk, build it, save it, and copy it
                print "****** Building *******\n\tfile: {}".format(filepath)
                print "\tfunc: {}".format(builder.__name__)
                print "*************"
                df = builder(*args, **kwargs)
                write(df, filepath)
                # Copy to Stata DTA if needed
                fileroot, fileext = splitext(filepath)
                if copydta and fileext != '.dta':
                    force_df(df).to_stata(fileroot + '.dta')

            return df
        return wrapper
    return actualDecorator


def load_or_build(filepath, force=False,
                  build=None, bargs=[], bkwargs=dict(),
                  copydta=False):
    """
    Loads `filepath` if it exists. If it does not exist, or if `force` is
    `True`, then builds a dataframe using `build` and writes it to disk at
    `filepath` for future access.

    Args
    -----
    `filepath`, str: path to DataFrame

    Kwargs
    -----
    `force`, bool (False): Build the DataFrame and save it to `filepath` even if
      `filepath` already exists.
    `build`, function (None): Function that returns a DataFrame to be saved at
      `filepath`
    `bargs`, list ([]): args to pass to `build` function.
    `bkwargs`, dict: kwargs to pass to `build` function.
    `copydta`, bool (False): If `filepath` is not a Stata file (.dta), also save
      the DataFrame as a Stata file by changing the extension of `filepath` to
      `.dta`

    Notes
    ------
    - If `filepath` does not exist and `build=None`, `IOError` is raised.
    - If `filepath` exists and `force=True`, `filepath` with be written over.
    """

    if isfile(filepath) and not force:
        return read(filepath)
    elif build is None:
        raise IOError("File doesn't exist:\n{}".format(filepath))
    else:
        print "****** Building *******\n\tfile: {}".format(filepath)
        print "\tfunc: {}".format(build.__name__)
        print "*************"
        df = build(*bargs, **bkwargs)
        write(df, filepath)

        fileroot, fileext = splitext(filepath)
        if copydta and fileext != '.dta':
            pd.DataFrame(df).to_stata(fileroot + '.dta')

        return df


def loadbuild_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--rebuild-down', action='store_true')
    parser.add_argument('--rebuild-all', action='store_true')
    args = parser.parse_args()

    rebuild = args.rebuild
    rebuild_down = args.rebuild_down
    rebuild_all = args.rebuild_all

    if rebuild_all:
        rebuild = True
        rebuild_down = True

    return rebuild, rebuild_down


def save_cli():
    """
    CLI option to `--save`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    return args.save


def try_pickle(filepath):
    """
    Use archived pickle for quicker reading. If archive doesn't exist,
    create it for next time.
    """

    fileroot, fileext = splitext(filepath)
    pickle_path = fileroot + '.p'

    if isfile(pickle_path):
        df = pd.read_pickle(pickle_path)
    else:
        df = read(filepath)
        df.to_pickle(pickle_path)

    return df


def read(path, **kwargs):
    """
    Read file to DataFrame by file's extension.

    Supported:
        csv
        p (pickle)
        hdf (HDF5)
        dta (Stata)
    """

    file_type = path.split('.')[-1]

    if file_type == 'csv':
        read_f = pd.read_csv
    elif file_type == 'p':
        read_f = pd.read_pickle
    elif file_type == 'hdf':
        read_f = pd.read_hdf
    elif file_type == 'dta':
        read_f = pd.read_stata
    else:
        err_str = 'File type {} is yet not supported'
        raise NotImplementedError(err_str.format(file_type))

    return read_f(path, **kwargs)


def write(df, path, **kwargs):
    """
    Write DataFrame to file by file's extension.

    Supported:
        csv
        p (pickle)
        hdf (HDF5)
        dta (Stata)
    """

    file_type = path.split('.')[-1]

    if file_type == 'csv':
        df.to_csv(path, **kwargs)
    elif file_type == 'p':
        df.to_pickle(path, **kwargs)
    elif file_type == 'hdf':
        df.to_hdf(path, 'frame', **kwargs)
    elif file_type == 'dta':
        df.to_stata(path, **kwargs)
    else:
        err_str = 'File type {} is yet not supported.'
        raise NotImplementedError(err_str.format(file_type))
