import pytest

from stanza import Pipeline
from stanza.tests import TEST_MODELS_DIR
from stanza.utils.datasets.coref.convert_hebrew_iahlt import extract_doc

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

@pytest.fixture(scope="module")
def tokenizer():
    pipe = Pipeline(lang="he", processors="tokenize", dir=TEST_MODELS_DIR, download_method=None)
    return pipe

TEXT = """



מבולבלים​? גם אנחנו​: ל​מסעדנים ו​ה​מלצרים יש עוד סימני שאלה על ה​טיפים​

ה​פער בין פסיקת בית ה​דין ל​עבודה לבין פסיקה קודמת של בג"ץ​, משאיר את ה​ענף ב​חוסר וודאות​, ו​ה -​1 ב​ינואר כבר מעבר ל​פינה . "​מ​בחינת​י , הייתי מוסיף ל​תפריט תוספת שירות של 17​% "​, אמר בעלים של מסעדה ב​שדרות​

ב​רשות ה​מיסים מסתפקים ב​מסר עמום באשר ל​כוונותי​הם לאור פסק דין ה​טיפים ש​צפוי להיכנס ל​תוקפ​ו ב​-​1 ב​ינואר . על פי פרשנות​ם ה​מקצועית , הבהירו​, יש מקום לחייב את כספי ה​טיפים ב​מע"מ , "​עם זאת​, ה​רשות עדין בוחנת את ה​סוגיה ו​טרם התקבלה החלטה אופרטיבית ב​עניין "​. ו​איך אמורים ה​מסעדנים להיערך בינתיים ל​יישום ה​פסיקה ו​ל​מחזור ה​שנה ה​באה ? ב​יום חמישי יפגשו אנשי ארגון '​מסעדנים חזקים ביחד​' עם מנהל רשות ה​מיסים ערן יעקב​, ו​ידרשו תשובות ברורות​.​

"​אני עדיין לא מדבר עם ה​עובדים של​י , ו​אני גם לא יודע איך להיערך החל מ​עוד שבועיים​"​, אמר ל​'​דבר ראשון​' ניר שוחט​, ה​בעלים של מסעדת סושי מוטו ב​שדרות ו​מוסיף כי יהיה קשה להתאים את ה​פסיקה ל​מציאות ב​שטח . "​אף אחד לא יודע​. יש המון סתירות – עורך ה​דין אומר דבר אחד ו​רואה ה​חשבון דבר אחר​. עדיין לא הצליחו להבין את ה​חוק ל​אשור​ו "​.​

"​מ​בחינת​י , הייתי מוסיף ל​תפריט תוספת שירות של 17​% . זה יגלם גם את ה​מע"מ ו​ה​טיפים ו​מ​זה אני אשלם ל​מלצרים . די כבר עם ה​טיפים ה​אלה , מספיק​.​"​
"""

CLUSTER = {'metadata': {'name': 'המסעדנים', 'entity': 'person'}, 'mentions': [[28, 35, {}], [572, 581, {}]]}

def test_extract_doc(tokenizer):
    doc = {'text': TEXT,
           'clusters': [CLUSTER],
           'metadata': {
               'doc_id': 'test'
           }
           }
    extracted = extract_doc(tokenizer, [doc])
    assert len(extracted) == 1
    assert len(extracted[0].coref_spans) == 2
    assert extracted[0].coref_spans[1] == [(0, 4, 4)]
    assert extracted[0].coref_spans[6] == [(0, 3, 4)]
