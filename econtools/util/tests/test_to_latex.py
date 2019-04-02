import pytest

import pandas as pd

from econtools.metrics.core import Results
from econtools.util.to_latex import (outreg, table_mainrow, table_statrow,
                                     _add_filler_empty_cells)


def results_factory(varnames, beta, se, pt, **kwargs):
        beta_s = pd.Series(beta, index=varnames)
        se_s = pd.Series(se, index=varnames)
        pt_s = pd.Series(pt, index=varnames)

        return Results(beta=beta_s, se=se_s, pt=pt_s)


class Test_outreg(object):

    def test_basic(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 2.02],
                               [0.005, 0.123])
        table_str = outreg((reg1, reg2),
                           ['x1'], ['Coeff 1'], digits=3)
        expected = (
            "Coeff 1  & 3.142**    & 3.111***    \\\\ \n"
            "         & (1.413)    & (1.413)     \\\\ \n"
        )
        assert table_str == expected

    def test_2vars(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 4.024],
                               [0.005, 0.123])
        table_str = outreg((reg1, reg2),
                           ['x1', 'x2'], ['Coeff 1', 'Coeff 2'], digits=3)
        expected = (
            "Coeff 1  & 3.142**    & 3.111***    \\\\ \n"
            "         & (1.413)    & (1.413)     \\\\ \n"
            "Coeff 2  & 1.590      & 1.390       \\\\ \n"
            "         & (2.020)    & (4.024)     \\\\ \n"
        )
        assert table_str == expected

    def test_3vars_1offset(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x3'],
                               [3.1111, 1.39],
                               [1.4134, 4.024],
                               [0.005, 0.123])
        table_str = outreg((reg1, reg2),
                           ['x1', 'x2', 'x3'],
                           ['Coeff 1', 'Coeff 2', 'Coeff 3'],
                           digits=3)
        expected = (
            "Coeff 1  & 3.142**    & 3.111***    \\\\ \n"
            "         & (1.413)    & (1.413)     \\\\ \n"
            "Coeff 2  & 1.590      &             \\\\ \n"
            "         & (2.020)    &             \\\\ \n"
            "Coeff 3  &            & 1.390       \\\\ \n"
            "         &            & (4.024)     \\\\ \n"
        )
        assert table_str == expected

    def test_no_vars(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 4.024],
                               [0.005, 0.123])
        table_str = outreg((reg1, reg2), digits=3)
        expected = (
            "x1  & 3.142**    & 3.111***    \\\\ \n"
            "    & (1.413)    & (1.413)     \\\\ \n"
            "x2  & 1.590      & 1.390       \\\\ \n"
            "    & (2.020)    & (4.024)     \\\\ \n"
        )
        assert table_str == expected

class Test_table_mainrow(object):

    def test_basic(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 2.02],
                               [0.005, 0.123])
        table_str = table_mainrow("Coeff 1", 'x1', (reg1, reg2), name_just=10,
                                  stat_just=10)
        expected = (
            "Coeff 1   & 3.142** & 3.111*** \\\\ \n"
            "          & (1.413) & (1.413)  \\\\ \n"
        )
        assert table_str == expected

    def test_missing_coeff(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x3', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 2.02],
                               [0.005, 0.123])
        table_str = table_mainrow("Coeff 1", 'x1', (reg1, reg2), name_just=10,
                                  stat_just=10)
        expected = (
            "Coeff 1   & 3.142** &          \\\\ \n"
            "          & (1.413) &          \\\\ \n"
        )
        assert table_str == expected

    def test_se_brackets(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 2.02],
                               [0.005, 0.123])
        table_str = table_mainrow("Coeff 1", 'x1', (reg1, reg2), name_just=10,
                                  stat_just=10, se="[")
        expected = (
            "Coeff 1   & 3.142** & 3.111*** \\\\ \n"
            "          & [1.413] & [1.413]  \\\\ \n"
        )
        assert table_str == expected

    def test_no_stars(self):
        reg1 = results_factory(['x1', 'x2'],
                               [3.14159, 1.59],
                               [1.41343, 2.02],
                               [0.035, 0.123])
        reg2 = results_factory(['x1', 'x2'],
                               [3.1111, 1.39],
                               [1.4134, 2.02],
                               [0.005, 0.123])
        table_str = table_mainrow("Coeff 1", 'x1', (reg1, reg2), name_just=10,
                                  stat_just=10, stars=False)
        expected = (
            "Coeff 1   & 3.142   & 3.111    \\\\ \n"
            "          & (1.413) & (1.413)  \\\\ \n"
        )
        assert table_str == expected


class Test_table_statrow(object):

    def test_std(self):
        vals = (1, 2, 3)
        rowname = 'Test'
        name_just = 6
        stat_just = 7
        digits = 2
        expected = (
            'Test  & 1.00 & 2.00 & 3.00  \\\\ \n'
        )
        result = table_statrow(rowname, vals, name_just=name_just,
                               stat_just=stat_just, digits=digits)
        assert expected == result

    def test_empty(self):
        vals = (1, '', 2, 3)
        rowname = 'Test'
        name_just = 6
        stat_just = 7
        digits = 2
        expected = (
            'Test  & 1.00 &      & 2.00 & 3.00  \\\\ \n'
        )
        result = table_statrow(rowname, vals, name_just=name_just,
                               stat_just=stat_just, digits=digits)
        assert expected == result

    def test_wrapnum(self):
        vals = (1, 2)
        rowname = 'Test'
        name_just = 6
        stat_just = 3
        digits = 2
        expected = (
            'Test  & \\num{1.00}& \\num{2.00} \\\\ \n'
        )
        result = table_statrow(rowname, vals, name_just=name_just,
                               stat_just=stat_just, digits=digits,
                               wrapnum=True)
        assert expected == result

    def test_sd(self):
        vals = (1, 2)
        rowname = 'Test'
        name_just = 6
        stat_just = 3
        digits = 2
        expected = (
            'Test  & (1.00)& (2.00) \\\\ \n'
        )
        result = table_statrow(rowname, vals, name_just=name_just,
                               stat_just=stat_just, digits=digits,
                               sd=True)
        assert expected == result

    def test_wrapnum_sd(self):
        vals = (1, 2)
        rowname = 'Test'
        name_just = 6
        stat_just = 3
        digits = 2
        expected = (
            'Test  & (\\num{1.00})& (\\num{2.00}) \\\\ \n'
        )
        result = table_statrow(rowname, vals, name_just=name_just,
                               stat_just=stat_just, digits=digits,
                               wrapnum=True, sd=True)
        assert expected == result


class Test_add_filler_empty_cells(object):

    def test_identity(self):
        orig_vals = (1, 2, 3)
        empty_left, empty_right, empty_slots = 0, 0, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert orig_vals == new_vals

    def test_left(self):
        orig_vals = (1, 2, 3)
        expected = ('', '', 1, 2, 3)
        empty_left, empty_right, empty_slots = 2, 0, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert expected == new_vals

    def test_right(self):
        orig_vals = (1, 2, 3)
        expected = (1, 2, 3, '', '', '')
        empty_left, empty_right, empty_slots = 0, 3, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert expected == new_vals

    def test_left_right(self):
        orig_vals = (1, 2, 3)
        expected = ('', '', 1, 2, 3, '', '', '')
        empty_left, empty_right, empty_slots = 2, 3, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert expected == new_vals

    def test_left_right_slots_error(self):
        orig_vals = (1, 2, 3)
        empty_left, empty_right, empty_slots = 2, 2, [2]
        with pytest.raises(ValueError):
            _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                    empty_slots)

    def test_empty_slots1(self):
        orig_vals = (1, 2, 3)
        expected = ('', '', 1, 2, 3, '', '', '')
        empty_left, empty_right = 0, 0
        empty_slots = (0, 1, 5, 6, 7)
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert expected == new_vals

    def test_empty_slots2(self):
        orig_vals = (1, 2, 3)
        expected = (1, '', 2, '', 3, '')
        empty_left, empty_right = 0, 0
        empty_slots = (1, 3, 5)
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert expected == new_vals

    def test_list_safe_in_memory(self):
        orig_vals = [1, 2, 3]
        expected_vals = [1, 2, 3]
        empty_left, empty_right = 0, 0
        empty_slots = (1, 3, 5)
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert orig_vals == expected_vals


if __name__ == '__main__':
    pass
