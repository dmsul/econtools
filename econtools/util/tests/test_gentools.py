import nose
from nose.tools import raises

from econtools.util import int2base, base2int


class Test_BaseConvert(object):

    def test_2b16_1(self):
        x = 17
        expected = '11'
        result = int2base(x, 16)
        assert expected == result

    def test_2b36_1(self):
        x = 37
        expected = '11'
        result = int2base(x, 36)
        assert expected == result

    def test_2b62_1(self):
        x = 63
        expected = '11'
        result = int2base(x, 62)
        assert expected == result

    def test_b16(self):
        base = 16
        x = 1022
        to = int2base(x, base)
        back = base2int(to, base)
        assert x == back

    def test_b36(self):
        base = 36
        x = 1022
        to = int2base(x, base)
        back = base2int(to, base)
        assert x == back

    def test_b62(self):
        base = 36
        x = 1022
        to = int2base(x, base)
        back = base2int(to, base)
        assert x == back

    @raises(ValueError)
    def test_invalid_char_b16(self):
        base2int('123Z', 16)

    @raises(ValueError)
    def test_invalid_char_b32(self):
        base2int('123a', 32)

    @raises(ValueError)
    def test_invalid_char_b62(self):
        base2int('123!', 62)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-v'])
