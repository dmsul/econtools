import os
from econtools.metrics.core import Results

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
                  digits=None):
    """
    Add a table row without standard errors.

    Args
    ----
    `rowname`, str: Row's name.
    `vals`, iterable: Values to fill cell rows. Can add empty cells with ''.

    Kwargs
    ------
    `name_just`, int (24): Number of characters to align rowname to.
    `stat_just`, int (12): Same.
    `wrapnum`, bool (False): If True, wrap cell values in LaTeX function `num`,
      which automatically adds commas as needed. Requires LaTex package
      `siunitx` to use in LaTeX.
    `sd`, bool (False): If True, wrap cell value in parentheses, as is often
      done when the value is a standard deviation.
    `digits`, int or None (None): How many digits after decimal to print. If
      `None`, prints contents of `vals` exactly as is.

    Return
    ------
    String of LaTeX tabular row with `rowname` and `vals` with the specified
      formatting.
    """

    outstr = rowname.ljust(name_just)

    if wrapnum:
        cell = "\\num{{{}}}"
    else:
        cell = "{}"

    if sd:
        cell = "(" + cell + ")"
    cell = "& " + cell

    if digits is None:
        def_printval = lambda x: x
    else:
        def_printval = lambda x: _format_nums(x, digits=digits)

    for val in vals:
        printval = def_printval(val)
        outstr += cell.format(printval).ljust(stat_just)

    outstr += eol

    return outstr


def table_mainrow(rowname, varname, regs,
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

    # Constants
    se_cell = "& [{}]"
    blank_stat = "& ".ljust(stat_just)
    # Start beta and SE rows
    beta_row = rowname.ljust(name_just)
    se_row = " ".ljust(name_just)
    for reg in regs:
        if type(reg) is not Results:
            beta_row += blank_stat
            se_row += blank_stat
        else:
            # Beta and stars
            this_beta = _format_nums(reg.beta[varname], digits=digits)
            this_sig = _sig_level(reg.pt[varname])
            beta_cell = "& {}".format(this_beta + this_sig)
            beta_row += beta_cell.ljust(stat_just)
            # Standard Error
            this_se = _format_nums(reg.se[varname], digits=digits)
            se_row += se_cell.format(this_se).ljust(stat_just)

    full_row = beta_row + eol + se_row + eol

    return full_row


def _format_nums(x, digits=3):
    if type(x) is str:
        return x
    else:
        return '{{:.{}f}}'.format(digits).format(x)


def _sig_level(p):
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
