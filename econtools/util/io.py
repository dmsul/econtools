import pandas as pd


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
