import os
from econtools.metrics.core import Results

eol = " \\\\ \n"
sig_labels = {1: '', .1: '*', .05: '**', .01: '***'}


# TODO: Add options for basic statrow (r2, N)? (how to handle 2sls r2?)
def outreg(regs, var_names, var_labels, digits=4, stars=True, se="(",
           options=False):
    opt_dict = _set_options(var_labels, digits, stars)
    table_str = ''
    for var_idx, varname in enumerate(var_names):
        table_str += table_mainrow(var_labels[var_idx], varname, regs,
                                   **opt_dict)

    if options:
        return table_str, opt_dict
    else:
        return table_str

def _set_options(var_labels, digits, stars):
    label_lens = [len(label) for label in var_labels]
    name_just = max(label_lens) + 2
    stat_just = (
        digits +
        3 +     # Leading zero, decimal, negative sign
        3 +     # Stars
        4       # Extra buffer
    )
    opt_dict = {
        'name_just': name_just,
        'stat_just': stat_just,
        'digits': digits,
        'stars': stars,
    }
    return opt_dict


def table_mainrow(rowname, varname, regs,
                  name_just=24, stat_just=12, digits=3, se="(",
                  stars=True):

    """
    Add a table row of regression coefficients with standard errors.

    Args
    ----
    `rowname`, str: First cell of table row, i.e., the row's name.
    `varname`, str: Name of variable to pull from metrics `Results`.
    `regs`, `Results` object or iterable of `Results`: Regressions from which
      to pull coefficients named `varname`.

    Kwargs
    ----
    `name_just`
    `stat_just`
    `digits`

    Returns
    ----
    String of table row.
    """

    # Start beta and SE rows
    beta_vals = []
    se_vals = []
    # Extract beta/sig and se values to pass to `table_statrow`
    for reg in regs:
        if type(reg) is not Results or varname not in reg.beta:
            beta_vals.append('')
            se_vals.append('')
        else:
            # Beta and stars
            this_beta = _format_nums(reg.beta[varname], digits=digits)
            if stars:
                this_sig = _sig_level(reg.pt[varname])
            else:
                this_sig = ''
            beta_vals.append(this_beta + this_sig)
            # Standard Error
            this_se = reg.se[varname]
            se_vals.append(this_se)

    beta_row = table_statrow(rowname, beta_vals, name_just=name_just,
                             stat_just=stat_just)
    se_row = table_statrow('', se_vals, name_just=name_just,
                           stat_just=stat_just, sd=se, digits=digits)

    full_row = beta_row + se_row

    return full_row


def table_statrow(rowname, vals, name_just=24, stat_just=12, wrapnum=False,
                  sd=False, digits=None,
                  empty_left=0, empty_right=0, empty_slots=[],
                  **kwargs):
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
      `siunitx` in LaTeX document.
    `sd`, bool/str (False): If True, wrap cell value in parentheses as per
      convention. May also set `sd="["` to wrap in brackets.
    `digits`, int or None (None): How many digits after decimal to print. If
      `None`, prints contents of `vals` exactly as is.
    `empty_left`, int (0): Adds `empty_left` empty cells to left side of row.
      Is mutually exclusive with `empty_slots`.
    `empty_right`, int (0): See `empty_left`.
    `empty_slots`, list ([]): Make table row have empty cells at index values
      in `empty_slots` (zero-indexed). Mutually exclusive with `empty_left` and
      `empty_right`. For example, passing `vals=(1, 2, 3)` and
      `empty_slots=(1, 3, 5)` is the same as passing
      `vals=(1, '', 2, '', 3, '')`.

    Return
    ------
    String of LaTeX tabular row with `rowname` and `vals` with the specified
      formatting.
    """

    outstr = rowname.ljust(name_just)

    cell = "\\num{{{}}}" if wrapnum else "{}"
    cell = _add_sd_parens(sd, cell)
    cell = "& " + cell

    vals = _add_filler_empty_cells(vals, empty_left, empty_right, empty_slots)

    for val in vals:
        # If empty string, add empty cell here (can't pass to `_format_nums` or
        # will get empty parens instead).
        if type(val) is str and len(val) == 0:
            outstr += "& ".ljust(stat_just)
        else:
            val_to_digits = (
                val if digits is None else _format_nums(val, digits=digits)
            )
            outstr += cell.format(val_to_digits).ljust(stat_just)

    outstr += eol

    return outstr

def _add_sd_parens(sd, cell):
    """ Wrap table cell in parens/brackets if needed """
    # Make `sd=True` same as `sd='('`
    if sd is True:
        sd = "("

    if type(sd) is str:
        # If `sd` is str, check if valid, then wrap `cell`
        if sd in ('(', '['):
            leftp = sd
            rightp = ")" if leftp == '(' else ']'
            cell = leftp + cell + rightp
        else:
            err_str = "Input '{}' invalid".format(sd)
            raise ValueError(err_str)
    elif sd is False:
        # If `sd` False, do nothing
        pass
    else:
        raise ValueError("Value `sd={}` invalid.".format(sd))

    return cell

def _add_filler_empty_cells(vals, empty_left, empty_right, empty_slots):
    # Convert left/right empty counts to list of empty slots
    if not (empty_left or empty_right or empty_slots):
        return vals
    elif (empty_left or empty_right) and empty_slots:
        raise ValueError("Cannot specify left/right empty and `empty_slots`.")
    elif not empty_slots:
        len_vals = len(vals)
        empty_slots = (
            list(range(empty_left)) +
            list(range(empty_left + len_vals,
                       empty_left + len_vals + empty_right))
        )

    # Add empty string to `vals` where appropriate
    new_vals = []
    nonempty_col = 0
    for i in range(len(vals) + len(empty_slots)):
        if i in empty_slots:
            new_vals.append('')
        else:
            new_vals.append(vals[nonempty_col])
            nonempty_col += 1

    return tuple(new_vals)


def _format_nums(x, digits=3):
    if type(x) is str:
        return x
    else:
        return '{{:.{}f}}'.format(digits).format(x)


# TODO: Make this adaptive
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


def write_notes(notes, table_path):
    split_path = os.path.splitext(table_path)
    notes_path = split_path[0] + '_notes.tex'
    with open(notes_path, 'w') as f:
        f.write(notes)
