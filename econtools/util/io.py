from os.path import isfile, splitext
from functools import wraps
import argparse

import pandas as pd

from .gentools import force_df

PICKLE_EXT = 'p'


# TODO: Rename `load_or_build` to `raw_load_or_build` (still want direct
# access), replace with decorator (once the usage in projects is updated and
# tested)
def dec_load_or_build(raw_filepath, copydta=False, path_args=[]):
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
    `_path`, list-like ([]): A list of `int`s or `str`s that point to args or
        kwargs of the build function, respectively. The value of these arguments
        will then be use to format `filepath`.
        Example:
        ```
        from econtools import dec_load_or_build

        @dec_load_or_build('file_{}_{}.csv')
        def foo(a, b=None, _path=[0, 'b']):
            return pd.DataFrame([a, b])

        if __name__ == '__main__':
            # Saves `df` to `file_infix_suffix.csv`
            foo('infix', 'suffix')
        ```
    """
    def actualDecorator(builder):
        @wraps(builder)
        def wrapper(*args, **kwargs):
            load = kwargs.pop('_load', True)
            rebuild = kwargs.pop('_rebuild', False)
            filepath = _set_filepath(raw_filepath, path_args, args, kwargs)
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

def _set_filepath(raw_path, path_args, args, kwargs):      #noqa
    format_filepath = _parse_pathargs(path_args, args, kwargs)
    try:
        filepath = raw_path.format(*format_filepath)
    except IndexError:
        err_str = (
            "Not enough arguments to fill file path template:\n"
            "    Args: {}\n"
            "    Path: {}"
        ).format(raw_path, format_filepath)
        raise ValueError(err_str)
    return filepath

def _parse_pathargs(path_args, args, kwargs):          #noqa
    """ Numbers are args, strings are kwargs.  """
    patharg_values = []

    for arg in path_args:
        arg_type = type(arg)
        if arg_type is int:
            patharg_values.append(args[arg])
        elif arg_type is str:
            patharg_values.append(kwargs[arg])
        else:
            raise ValueError("Path arg must be int or str.")

    return patharg_values


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
    """ Convenience CLI args for rebuilding data using `load_or_build` """
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
    """ CLI option to `--save` """
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
    pickle_path = fileroot + '.' + PICKLE_EXT

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
    elif file_type == PICKLE_EXT:
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
    elif file_type == PICKLE_EXT:
        df.to_pickle(path, **kwargs)
    elif file_type == 'hdf':
        df.to_hdf(path, 'frame', **kwargs)
    elif file_type == 'dta':
        df.to_stata(path, **kwargs)
    else:
        err_str = 'File type {} is yet not supported.'
        raise NotImplementedError(err_str.format(file_type))
