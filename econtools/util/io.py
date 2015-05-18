from os.path import isfile, splitext

import pandas as pd


def load_or_build(filepath, force=False,
                  build=None, bargs=[], bkwargs=dict(),
                  copydta=False):
    """Load `filepath` or build if necessary using function `build`"""

    if isfile(filepath) and not force:
        return read(filepath)
    elif build is None:
        raise IOError("File doesn't exist:\n{}".format(filepath))
    else:
        df = build(*bargs, **bkwargs)
        write(df, filepath)

        fileroot, fileext = splitext(filepath)
        if copydta and fileext != '.dta':
            pd.DataFrame(df).to_stata(fileroot + '.dta')

        return df


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
        dta (Stata)
    """

    file_type = path.split('.')[-1]

    if file_type == 'csv':
        read_f = pd.read_csv
    elif file_type == 'p':
        read_f = pd.read_pickle
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
        dta (Stata)
    """

    file_type = path.split('.')[-1]

    if file_type == 'csv':
        df.to_csv(path, **kwargs)
    elif file_type == 'p':
        df.to_pickle(path, **kwargs)
    elif file_type == 'dta':
        df.to_stata(path, **kwargs)
    else:
        err_str = 'File type {} is yet not supported.'
        raise NotImplementedError(err_str.format(file_type))
