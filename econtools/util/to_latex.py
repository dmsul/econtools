import os
from econtools.util import force_iterable

eol = " \\\\ \n"
sig_labels = {1: '', .1: '*', .05: '**', .01: '***'}


class table(object):
    """
    Wrapper for table row functions that stores justification and `digits`
      values so you don't have to repeat them every tiime.
    """

    def __init__(self, name_just=24, stat_just=12, digits=3):
        self.name_just = name_just
        self.stat_just = stat_just
        self.digits = digits
        self.check_val = ('name_just', 'stat_just', 'digits')

    def statrow(self, *args, **kwargs):
        for val in self.check_val:
            if val not in kwargs:
                kwargs[val] = self.__dict__[val]

        return table_statrow(*args, **kwargs)

    def mainrow(self, *args, **kwargs):
        for val in self.check_val:
            if val not in kwargs:
                kwargs[val] = self.__dict__[val]

        return table_mainrow(*args, **kwargs)


def table_statrow(rowname, vals, name_just=24, stat_just=12, wrapnum=False,
                  sd=False,
                  digits=None,
                  empty=[]):
    """
    Add a table row without standard errors.

    `digits` must be specified for numerical values, otherwise assumes string.
    """
    outstr = rowname.ljust(name_just)

    if wrapnum:
        cell = "\\num{{{}}}"
    else:
        cell = "{}"
    if sd:
        cell = "(" + cell + ")"
    cell = "& " + cell

    for i, val in enumerate(vals):

        if i in empty:
            outstr += "& ".ljust(stat_just)
            continue

        if digits is not None:
            printval = _format_nums(val, digits=digits)
        else:
            printval = val
        outstr += cell.format(printval).ljust(stat_just)

    # Add right-hand empty cells if needed
    max_val_index = len(vals) - 1
    if len(empty) > 0 and (max(empty) > max_val_index):
        outstr += "& ".ljust(stat_just)*(max(empty) - max_val_index)

    outstr += eol
    return outstr


def table_mainrow(rowname, varname, regs,
                  lempty=0, rempty=0, empty=[],
                  name_just=24, stat_just=12, digits=3):

    """
    Add a table row of regression coefficients with standard errors.

    Args
    ----
    `rowname`, str: First cell of table row, i.e., the row's name.
    `varname`, str: Name of variable to pull from metrics `Results`.
    `regs`, `Results` object or iterable of `Results`: Regressions from which to
      pull coefficients named `varname`.

    Kwargs
    ----
    `lempty`
    `rempty`
    `empty`
    `name_just`
    `stat_just`
    `digits`

    Returns
    ----
    String of table row.
    """

    # Translate old `lempty` into `empty` list
    len_vals = len(force_iterable(regs))
    if (lempty or rempty) and empty:
        raise ValueError
    elif not empty:
        empty = range(lempty) + range(lempty + len_vals, len_vals + rempty)
    len_empty = len(empty)
    len_row = len_empty + len_vals

    # Constants
    se_cell = "& [{}]"
    blank_stat = "& ".ljust(stat_just)
    # Build beta and SE rows
    beta_row = rowname.ljust(name_just)
    se_row = " ".ljust(name_just)
    nonempty_col = 0
    for i in range(len_row):
        if i in empty:
            beta_row += blank_stat
            se_row += blank_stat
        else:
            stats = _get_stats(force_iterable(regs)[nonempty_col],
                               varname, '', digits)
            this_beta = "& {}".format(stats['_beta'] + stats['_sig'])
            beta_row += this_beta.ljust(stat_just)
            se_row += se_cell.format(stats['_se']).ljust(stat_just)
            nonempty_col += 1
    assert nonempty_col == len_vals

    full_row = beta_row + eol + se_row + eol

    return full_row

def _get_stats(reg, varname, label, digits=3):        #noqa
    beta = _format_nums(reg.beta[varname], digits=digits)
    se = _format_nums(reg.se[varname], digits=digits)
    sig = _sig_level(reg.pt[varname])
    names = ['beta', 'sig', 'se']
    stats_dict = dict(zip(
        ['{}_{}'.format(label, x) for x in names],
        (beta, sig, se)
    ))
    return stats_dict

def _format_nums(x, digits=3):        #noqa
    if type(x) is str:
        return x
    else:
        return '{{:.{}f}}'.format(digits).format(x)

def _sig_level(p):      #noqa
    if p > .1:
        p_level = 1
    elif .05 < p <= .1:
        p_level = .1
    elif .01 < p <= .05:
        p_level = .05
    else:
        p_level = .01

    return sig_labels[p_level]


def join_latex_rows(row1, row2):
    """
    Assumes both end with `eol` and first column is label.
    """
    row1_noend = row1.replace(eol, "")
    row2_guts = row2.split("&", 1)[1:].replace(eol, "")
    joined = row1_noend + row2_guts
    return joined


def write_notes(notes, table_path):
    split_path = os.path.splitext(table_path)
    notes_path = split_path[0] + '_notes.tex'
    with open(notes_path, 'w') as f:
        f.write(notes)
