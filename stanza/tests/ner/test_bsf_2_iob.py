"""
Tests the conversion code for the lang_uk NER dataset
"""

import unittest
from stanza.utils.datasets.ner.convert_bsf_to_beios import convert_bsf, parse_bsf, BsfInfo

import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestBsf2Iob(unittest.TestCase):

    def test_1line_follow_markup_iob(self):
        data = 'тележурналіст Василь .'
        bsf_markup = 'T1	PERS 14 20	Василь'
        expected = '''тележурналіст O
Василь B-PERS
. O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_1line_2tok_markup_iob(self):
        data = 'тележурналіст Василь Нагірний .'
        bsf_markup = 'T1	PERS 14 29	Василь Нагірний'
        expected = '''тележурналіст O
Василь B-PERS
Нагірний I-PERS
. O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_1line_Long_tok_markup_iob(self):
        data = 'А в музеї Гуцульщини і Покуття можна '
        bsf_markup = 'T12	ORG 4 30	музеї Гуцульщини і Покуття'
        expected = '''А O
в O
музеї B-ORG
Гуцульщини I-ORG
і I-ORG
Покуття I-ORG
можна O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_2line_2tok_markup_iob(self):
        data = '''тележурналіст Василь Нагірний .
В івано-франківському видавництві «Лілея НВ» вийшла друком'''
        bsf_markup = '''T1	PERS 14 29	Василь Нагірний
T2	ORG 67 75	Лілея НВ'''
        expected = '''тележурналіст O
Василь B-PERS
Нагірний I-PERS
. O


В O
івано-франківському O
видавництві O
« O
Лілея B-ORG
НВ I-ORG
» O
вийшла O
друком O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_all_multiline_iob(self):
        data = '''його книжечка «А .
Kubler .
Світло і тіні маестро» .
Причому'''
        bsf_markup = '''T4	MISC 15 49	А .
Kubler .
Світло і тіні маестро
'''
        expected = '''його O
книжечка O
« O
А B-MISC
. I-MISC
Kubler I-MISC
. I-MISC
Світло I-MISC
і I-MISC
тіні I-MISC
маестро I-MISC
» O
. O


Причому O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))


if __name__ == '__main__':
    unittest.main()
