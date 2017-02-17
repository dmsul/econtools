import nose
from nose.tools import assert_equal, raises

from econtools.util.to_latex import table_statrow, _add_filler_empty_cells


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
        assert_equal(expected, result)

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
        assert_equal(expected, result)

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
        assert_equal(expected, result)

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
        assert_equal(expected, result)

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
        assert_equal(expected, result)


class Test_add_filler_empty_cells(object):

    def test_identity(self):
        orig_vals = (1, 2, 3)
        empty_left, empty_right, empty_slots = 0, 0, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert_equal(orig_vals, new_vals)

    def test_left(self):
        orig_vals = (1, 2, 3)
        expected = ('', '', 1, 2, 3)
        empty_left, empty_right, empty_slots = 2, 0, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert_equal(expected, new_vals)

    def test_right(self):
        orig_vals = (1, 2, 3)
        expected = (1, 2, 3, '', '', '')
        empty_left, empty_right, empty_slots = 0, 3, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert_equal(expected, new_vals)

    def test_left_right(self):
        orig_vals = (1, 2, 3)
        expected = ('', '', 1, 2, 3, '', '', '')
        empty_left, empty_right, empty_slots = 2, 3, []
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert_equal(expected, new_vals)

    @raises(ValueError)
    def test_left_right_slots_error(self):
        orig_vals = (1, 2, 3)
        empty_left, empty_right, empty_slots = 2, 2, [2]
        _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                empty_slots)

    def test_empty_slots1(self):
        orig_vals = (1, 2, 3)
        expected = ('', '', 1, 2, 3, '', '', '')
        empty_left, empty_right = 0, 0
        empty_slots = (0, 1, 5, 6, 7)
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert_equal(expected, new_vals)

    def test_empty_slots2(self):
        orig_vals = (1, 2, 3)
        expected = (1, '', 2, '', 3, '')
        empty_left, empty_right = 0, 0
        empty_slots = (1, 3, 5)
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,
                                           empty_slots)
        assert_equal(expected, new_vals)

    def test_list_safe_in_memory(self):
        orig_vals = [1, 2, 3]
        expected_vals = [1, 2, 3]
        empty_left, empty_right = 0, 0
        empty_slots = (1, 3, 5)
        new_vals = _add_filler_empty_cells(orig_vals, empty_left, empty_right,  # noqa
                                           empty_slots)
        assert_equal(orig_vals, expected_vals)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-v'])
