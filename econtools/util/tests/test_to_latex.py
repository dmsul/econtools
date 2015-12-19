import nose
from nose.tools import assert_equal

from econtools.util.to_latex import table_statrow


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


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-v'])
