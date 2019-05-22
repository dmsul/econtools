import argparse
from datetime import datetime
import re
import warnings
from functools import wraps
from inspect import getfullargspec
from os.path import isfile, splitext
from typing import Optional, Callable, List, Tuple

import numpy as np
import pandas as pd

from econtools.util.frametools import df_to_list


PICKLE_EXT = ('pkl', 'p')   # First is default for writing to pickle
HDF5_EXT = ('h5', 'hdf5')


def load_or_build(raw_filepath: str,
                  copydta: bool = False,
                  path_args: list = []) -> Callable:
    """Loads `raw_filepath` as a DataFrame if it exists, otherwise builds the
    data and saves it to `raw_filepath`.

    Args:
        raw_filepath (str): Path to saved DataFrame. If ``raw_filepath``
            includes named replacement fields (e.g., "``'{arg_name}'``") with
            the same name as function arguments, passed values will be inserted
            into the file path.

    Example:

        .. code-block:: python

            @load_or_build('data_for_{year}.pkl')
            def make_data(year):
                <Make the data>

            df = make_data(2018)    # Saves to 'data_for_2018.pkl'

    Keyword Args:
        copydta (bool): Defaults to False. If true, save a copy of the data in
            Stata DTA format if ``raw_filepath`` is not already a DTA file.

        path_args (list-like): DEPRECATED: Use named replacement fields
            instead. A list of ints or strs that point to args or
            kwargs of the build function, respectively. The value of these
            arguments will then be use to format ``raw_filepath``.

    Example:

        .. code-block:: python

            @load_or_build('file_{}_{}.csv', path_args=[0, 'b'])
            def foo(a, b=None):
                return pd.DataFrame([a, b])

            if __name__ == '__main__':
                # Saves `df` to `file_infix_suffix.csv`
                foo('infix', 'suffix')

    Other Parameters:
        These are additional kwargs that can be passed to the wrapped function
            that affect the behavior of ``load_or_build``.

        _rebuild (bool): Defaults to False. If True, build the DataFrame and
            save it to `filepath` even if `filepath` already exists.
        _load (bool): Defaults to True. If True, try loading the data before
            building it. If False, the building function is called and the
            result returned with no data written to disk.
    """
    def actualDecorator(builder):
        @wraps(builder)
        def wrapper(*args, **kwargs):
            # Filter the kwargs from `builder` meant for `load_or_build`
            load = kwargs.pop('_load', True)
            rebuild = kwargs.pop('_rebuild', False)

            # Format the file path
            if not path_args:   # XXX this is preferred/keep old for compat
                filepath = _set_filepath(raw_filepath, args, kwargs, builder)
            else:
                warnings.warn(DeprecationWarning(
                    "`path_args` is deprecated. Use curly brackets {}."))
                filepath = _set_filepath_old(raw_filepath, path_args, args,
                                             kwargs, builder)

            if load and isfile(filepath) and not rebuild:
                # If it's on disk and there are no objections, read it
                df = read(filepath)
            elif not load:
                # If `load = False` is passed, just return the built `df`
                df = builder(*args, **kwargs)
            else:
                # If it's just not on disk, build it, save it, and copy it
                start_line = "BUILDING @ {} {} using {}"  # Time, file, func
                finish_line = "WRITTEN  @ {} {}"          # Time, file
                func_name = builder.__module__ + ':' + builder.__name__
                print(start_line.format(_now(), filepath, func_name))
                df = builder(*args, **kwargs)             # Build it!
                write(df, filepath)                       # Save it!
                print(finish_line.format(_now(), filepath), flush=True)

                # Copy to Stata DTA if needed
                if copydta and filepath[-4:] != '.dta':
                    df.to_stata(filepath[:-4] + '.dta')

            return df
        return wrapper
    return actualDecorator

def _set_filepath(raw_filepath: str,
                  args: list,
                  kwargs: dict,
                  builder: Callable) -> str:
    argspec = getfullargspec(builder)
    if re.search('{.*}', raw_filepath):
        argspec = getfullargspec(builder)
        arg_names = argspec.args
        arg_dict = dict()
        if argspec.defaults:
            kwarg_defaults = list(argspec.defaults)
            while kwarg_defaults:
                arg_dict[arg_names.pop()] = kwarg_defaults.pop()

        for idx, arg in enumerate(args):
            arg_dict[argspec.args[idx]] = arg
        for k, v in kwargs.items():
            arg_dict[k] = v
        new_filepath = raw_filepath.format(**arg_dict)
    else:
        new_filepath = raw_filepath

    return new_filepath

def _set_filepath_old(raw_path, path_args, args, kwargs, builder):
    """
    Parse `args` and `kwargs` as directed by `path_args` and insert them
    into `raw_path`.
    """
    argspec = getfullargspec(builder)
    format_filepath = _parse_pathargs(path_args, args, kwargs, argspec)
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

def _parse_pathargs(path_args, args, kwargs, argspec):
    """
    `path_args`, iterable: Which args/kwargs to use, in order. Numbers are
        args, strings are kwargs.
    `args`, `kwargs`: The value of args/kwargs passed to the original builder
        function.
    `argspec`: Stores values of default kwargs from builder function.
    """
    patharg_values = []
    # Handle default kwargs
    argnames = argspec.args
    defaults = argspec.defaults

    for arg in path_args:
        arg_type = type(arg)
        if arg_type is int:
            patharg_values.append(args[arg])
        elif arg_type is str:
            try:
                patharg_values.append(kwargs[arg])
            except KeyError:
                nargs = len(argnames) - len(defaults)
                kwarg_idx = argnames.index(arg) - nargs
                patharg_values.append(defaults[kwarg_idx])
        else:
            raise ValueError("Path arg must be int or str.")

    return patharg_values

def _now():
    return datetime.now().strftime('%H:%M:%S')


def load_or_build_direct(filepath: str, force: bool=False,
                         build: Callable=None, bargs: list=[],
                         bkwargs: dict=dict(),
                         copydta: bool=False) -> pd.DataFrame:
    """
    Loads `filepath` if it exists. If it does not exist, or if `force` is
    `True`, then builds a dataframe using `build` and writes it to disk at
    `filepath` for future access.

    Args
    -----
    `filepath`, str: path to DataFrame

    Kwargs
    -----
    `force`, bool (False): Build the DataFrame and save it to `filepath` even
      if `filepath` already exists.
    `build`, function (None): Function that returns a DataFrame to be saved at
      `filepath`
    `bargs`, list ([]): args to pass to `build` function.
    `bkwargs`, dict: kwargs to pass to `build` function.
    `copydta`, bool (False): If `filepath` is not a Stata file (.dta), also
    save the DataFrame as a Stata file by changing the extension of `filepath`
      to `.dta`

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
        print("****** Building *******\n\tfile: {}".format(filepath))
        print("\tfunc: {}".format(build.__name__))
        print("*************")
        df = build(*bargs, **bkwargs)
        write(df, filepath)

        fileroot, fileext = splitext(filepath)
        if copydta and fileext != '.dta':
            pd.DataFrame(df).to_stata(fileroot + '.dta')

        return df


def loadbuild_cli() -> Tuple[bool, bool]:
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


def save_cli() -> bool:
    """Add CLI boolean flag ``--save``

    Returns:
        bool: `True` if ``--save`` was entered on command line, else `False`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    return args.save


def try_pickle(filepath: str) -> pd.DataFrame:
    """
    Use archived pickle for quicker reading. If archive doesn't exist,
    create it for next time.
    """

    fileroot, fileext = splitext(filepath)
    pickle_path = fileroot + '.' + PICKLE_EXT[0]

    if isfile(pickle_path):
        df = pd.read_pickle(pickle_path)
    else:
        df = read(filepath)
        df.to_pickle(pickle_path)

    return df


def read(path: str, **kwargs) -> pd.DataFrame:
    """Read file to DataFrame by file's extension.

    Args:
        path (str): Path to read the file from. Supported file suffixes are:
            - csv
            - pkl (pickle)
            - hdf (HDF5)
            - dta (Stata)
        **kwargs: Arbitrary keyword arguments to pass to the ``pandas`` read
            method.
    Returns:
        DataFrame:
    """

    file_type = path.split('.')[-1]

    if file_type == 'csv':
        read_f = pd.read_csv
    elif file_type in PICKLE_EXT:
        read_f = pd.read_pickle
    elif file_type in HDF5_EXT:
        read_f = pd.read_hdf
    elif file_type == 'dta':
        read_f = pd.read_stata
    else:
        err_str = 'File type {} is yet not supported'
        raise NotImplementedError(err_str.format(file_type))

    return read_f(path, **kwargs)


def write(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Read file to DataFrame by file's extension.

    Args:
        df (DataFrame): DataFrame to write to disk.
        path (str): Path to write the file to. Supported file suffixes are:
            - csv
            - pkl (pickle)
            - hdf (HDF5)
            - dta (Stata)
        **kwargs: Arbitrary keyword arguments to pass to the ``pandas`` write
            method.
    Returns:
        None:
    """

    file_type = path.split('.')[-1]

    if file_type == 'csv':
        df.to_csv(path, **kwargs)
    elif file_type in PICKLE_EXT:
        df.to_pickle(path, **kwargs)
    elif file_type in HDF5_EXT:
        mode = kwargs.pop('mode', 'w')
        df.to_hdf(path, 'df', mode=mode, **kwargs)
    elif file_type == 'dta':
        df.to_stata(path, **kwargs)
    else:
        err_str = 'File type {} is yet not supported.'
        raise NotImplementedError(err_str.format(file_type))


# Iteractive stuff
def confirmer(prompt_str: str, default_no: bool = True) -> bool:
    """Prompt user for yes/no answer.

    Args:
        prompt_str (str): Prompt to show user.
        default_no (bool): Defaults to True. If True, the default response is
            'No'.

    returns:
        bool: `True` if user responded 'Yes', else `False`.
    """
    yes_opts = ['Y', 'y', 'yes', 'Yes', 'YES']
    no_opts = ['N', 'n', 'no', 'No', 'NO']
    default_opt = ('',)
    if default_no:
        choices = ' (y/[n]) >>> '
        no_opts += default_opt
    else:
        choices = ' ([y]/n) >>> '
        yes_opts += default_opt

    full_prompt = prompt_str + choices

    ans = force_valid_response(full_prompt, yes_opts + no_opts)

    return ans in yes_opts


def force_valid_response(prompt_str: str, good_answers: List[str],
                         listin: bool = False, dtype=None,
                         _count: int = 0) -> str:

    ans = input(prompt_str)

    if listin:
        output = _parse_list_input(ans, dtype)
        set_ans = set(output)
        good = set_ans <= set(good_answers)
    else:
        output = ans
        good = ans in good_answers

    if not good and _count < 4:
        print("Invalid Response '{}'!".format(ans))
        _count += 1
        output = force_valid_response(prompt_str, good_answers, listin=listin,
                                      dtype=dtype, _count=_count)
    elif not good:
        raise ValueError('Learn to read')

    return output

def _parse_list_input(inp, dtype):
    inp = re.sub(r'\s\s+', ' ', inp)
    list_inp = inp.split(' ')
    if dtype:
        list_inp = map(dtype, list_inp)
    return list_inp


class DataInteractModel(object):

    def __init__(self, looplist, **kwargs):
        self.looplist = df_to_list(looplist)
        self.__dict__.update(kwargs)  # For secondary DataFrames

    def interact(self,
                 filepath: Optional[str] = None,
                 writeargs: dict = dict()) -> pd.DataFrame:

        if filepath:
            split_path = splitext(filepath)
            # Write notes in same format
            notes_path = split_path[0] + '_notes' + split_path[1]
            # Write log in ascii
            log_path = split_path[0] + '_log.txt'
            if isfile(log_path):
                overwrite = confirmer('Log file already exists. Overwrite?')
                if not overwrite:
                    import sys
                    sys.exit(0)

        responses = []
        notes = []
        while self.looplist:
            result, notes_on_result = self.display(self.looplist.pop())
            if result is not None:
                responses.append(result)
                notes.append(notes_on_result)

        responses_df = pd.DataFrame(responses)
        notes_df = pd.DataFrame(notes)

        if filepath:
            write(responses_df, filepath, **writeargs)
            write(notes_df, notes_path, **writeargs)
            self.write_log(log_path, responses_df, notes_df)

        return responses_df

    def display(self, row):
        """
        Make a prompt string, define valid input, define response to input.
        Should ultimately return two Series: the main result and notes.
        """
        pass

    def write_log(self,
                  log_path: str,
                  outdf: pd.DataFrame,
                  notes: pd.DataFrame) -> None:
        """
        By default writes the DataFrame as a dictionary for easy pasting
        into code. Can be overridden.
        Default format: idx: (result, notes),
        """

        with open(log_path, 'w') as f:
            f.write('Columns: {}\n\n'.format(outdf.columns.values))
            for idx, row in outdf.iterrows():
                key_str = "'{}':\t" if type(idx) is str else "{}: "
                full_line = key_str + "({}, '{}'),\n"
                # Fix float/int/nan crap
                row_list = _fix_dtypes(row.tolist())
                f.write(
                    full_line.format(idx, row_list, notes.loc[idx].squeeze()))

    def _force_valid_response(self, *args, **kwargs):
        return force_valid_response(*args, **kwargs)


def _fix_dtypes(rowlist: list) -> list:
    newlist = []
    for x in rowlist:
        dtype = type(x)
        if issubclass(dtype, np.floating) or dtype is float:
            if np.isnan(x):
                continue
            elif int(x) == x:
                newlist.append(int(x))
                continue
        newlist.append(x)
    return newlist
