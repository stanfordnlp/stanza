"""
Tests the conversion code for the lang_uk NER dataset
"""

import unittest
from stanza.utils.datasets.ner.convert_bsf_to_beios import convert_bsf, parse_bsf, BsfInfo

import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestBsf2Beios(unittest.TestCase):
    
    def test_empty_markup(self):
        res = convert_bsf('', '')
        self.assertEqual('', res)

    def test_1line_markup(self):
        data = 'тележурналіст Василь'
        bsf_markup = 'T1	PERS 14 20	Василь'
        expected = '''тележурналіст O
Василь S-PERS'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_1line_follow_markup(self):
        data = 'тележурналіст Василь .'
        bsf_markup = 'T1	PERS 14 20	Василь'
        expected = '''тележурналіст O
Василь S-PERS
. O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_1line_2tok_markup(self):
        data = 'тележурналіст Василь Нагірний .'
        bsf_markup = 'T1	PERS 14 29	Василь Нагірний'
        expected = '''тележурналіст O
Василь B-PERS
Нагірний E-PERS
. O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_1line_Long_tok_markup(self):
        data = 'А в музеї Гуцульщини і Покуття можна '
        bsf_markup = 'T12	ORG 4 30	музеї Гуцульщини і Покуття'
        expected = '''А O
в O
музеї B-ORG
Гуцульщини I-ORG
і I-ORG
Покуття E-ORG
можна O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_2line_2tok_markup(self):
        data = '''тележурналіст Василь Нагірний .
В івано-франківському видавництві «Лілея НВ» вийшла друком'''
        bsf_markup = '''T1	PERS 14 29	Василь Нагірний
T2	ORG 67 75	Лілея НВ'''
        expected = '''тележурналіст O
Василь B-PERS
Нагірний E-PERS
. O
В O
івано-франківському O
видавництві O
« O
Лілея B-ORG
НВ E-ORG
» O
вийшла O
друком O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_real_markup(self):
        data = '''Через напіввоєнний стан в Україні та збільшення телефонних терористичних погроз українці купуватимуть sim-карти тільки за паспортами .
Про це повідомив начальник управління зв'язків зі ЗМІ адміністрації Держспецзв'язку Віталій Кукса .
Він зауважив , що днями відомство опублікує проект змін до правил надання телекомунікаційних послуг , де будуть прописані норми ідентифікації громадян .
Абонентів , які на сьогодні вже мають sim-карту , за словами Віталія Кукси , реєструватимуть , коли ті звертатимуться в службу підтримки свого оператора мобільного зв'язку .
Однак мобільні оператори побоюються , що таке нововведення помітно зменшить продаж стартових пакетів , адже спеціалізовані магазини є лише у містах .
Відтак купити сімку в невеликих населених пунктах буде неможливо .
Крім того , нова процедура ідентифікації абонентів вимагатиме від операторів мобільного зв'язку додаткових витрат .
- Близько 90 % українських абонентів - це абоненти передоплати .
Якщо мова буде йти навіть про поетапну їх ідентифікацію , зробити це буде складно , довго і дорого .
Мобільним операторам доведеться йти на чималі витрати , пов'язані з укладанням і зберіганням договорів , веденням баз даних , - розповіла « Економічній правді » начальник відділу зв'язків з громадськістю « МТС-Україна » Вікторія Рубан .
'''
        bsf_markup = '''T1	LOC 26 33	Україні
T2	ORG 203 218	Держспецзв'язку
T3	PERS 219 232	Віталій Кукса
T4	PERS 449 462	Віталія Кукси
T5	ORG 1201 1219	Економічній правді
T6	ORG 1267 1278	МТС-Україна
T7	PERS 1281 1295	Вікторія Рубан
'''
        expected = '''Через O
напіввоєнний O
стан O
в O
Україні S-LOC
та O
збільшення O
телефонних O
терористичних O
погроз O
українці O
купуватимуть O
sim-карти O
тільки O
за O
паспортами O
. O
Про O
це O
повідомив O
начальник O
управління O
зв'язків O
зі O
ЗМІ O
адміністрації O
Держспецзв'язку S-ORG
Віталій B-PERS
Кукса E-PERS
. O
Він O
зауважив O
, O
що O
днями O
відомство O
опублікує O
проект O
змін O
до O
правил O
надання O
телекомунікаційних O
послуг O
, O
де O
будуть O
прописані O
норми O
ідентифікації O
громадян O
. O
Абонентів O
, O
які O
на O
сьогодні O
вже O
мають O
sim-карту O
, O
за O
словами O
Віталія B-PERS
Кукси E-PERS
, O
реєструватимуть O
, O
коли O
ті O
звертатимуться O
в O
службу O
підтримки O
свого O
оператора O
мобільного O
зв'язку O
. O
Однак O
мобільні O
оператори O
побоюються O
, O
що O
таке O
нововведення O
помітно O
зменшить O
продаж O
стартових O
пакетів O
, O
адже O
спеціалізовані O
магазини O
є O
лише O
у O
містах O
. O
Відтак O
купити O
сімку O
в O
невеликих O
населених O
пунктах O
буде O
неможливо O
. O
Крім O
того O
, O
нова O
процедура O
ідентифікації O
абонентів O
вимагатиме O
від O
операторів O
мобільного O
зв'язку O
додаткових O
витрат O
. O
- O
Близько O
90 O
% O
українських O
абонентів O
- O
це O
абоненти O
передоплати O
. O
Якщо O
мова O
буде O
йти O
навіть O
про O
поетапну O
їх O
ідентифікацію O
, O
зробити O
це O
буде O
складно O
, O
довго O
і O
дорого O
. O
Мобільним O
операторам O
доведеться O
йти O
на O
чималі O
витрати O
, O
пов'язані O
з O
укладанням O
і O
зберіганням O
договорів O
, O
веденням O
баз O
даних O
, O
- O
розповіла O
« O
Економічній B-ORG
правді E-ORG
» O
начальник O
відділу O
зв'язків O
з O
громадськістю O
« O
МТС-Україна S-ORG
» O
Вікторія B-PERS
Рубан E-PERS
. O'''
        self.assertEqual(expected, convert_bsf(data, bsf_markup))


class TestBsf(unittest.TestCase):

    def test_empty_bsf(self):
        self.assertEqual(parse_bsf(''), [])

    def test_empty2_bsf(self):
        self.assertEqual(parse_bsf(' \n \n'), [])

    def test_1line_bsf(self):
        bsf = 'T1	PERS 103 118	Василь Нагірний'
        res = parse_bsf(bsf)
        expected = BsfInfo('T1', 'PERS', 103, 118, 'Василь Нагірний')
        self.assertEqual(len(res), 1)
        self.assertEqual(res, [expected])

    def test_2line_bsf(self):
        bsf = '''T9	PERS 778 783	Карла
T10	MISC 814 819	міста'''
        res = parse_bsf(bsf)
        expected = [BsfInfo('T9', 'PERS', 778, 783, 'Карла'),
                    BsfInfo('T10', 'MISC', 814, 819, 'міста')]
        self.assertEqual(len(res), 2)
        self.assertEqual(res, expected)

    def test_multiline_bsf(self):
        bsf = '''T3	PERS 220 235	Андрієм Кіщуком
T4	MISC 251 285	А .
Kubler .
Світло і тіні маестро
T5	PERS 363 369	Кіблер'''
        res = parse_bsf(bsf)
        expected = [BsfInfo('T3', 'PERS', 220, 235, 'Андрієм Кіщуком'),
                    BsfInfo('T4', 'MISC', 251, 285, '''А .
Kubler .
Світло і тіні маестро'''),
                    BsfInfo('T5', 'PERS', 363, 369, 'Кіблер')]
        self.assertEqual(len(res), len(expected))
        self.assertEqual(res, expected)


if __name__ == '__main__':
    unittest.main()
