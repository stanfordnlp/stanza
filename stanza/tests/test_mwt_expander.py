"""
Basic testing of multi-word-token expansion
"""

import pytest
import stanza

from stanza.tests import *

pytestmark = pytest.mark.pipeline

# mwt data for testing
FR_MWT_SENTENCE = "Alors encore inconnu du grand public, Emmanuel Macron devient en 2014 ministre de l'Économie, de " \
                  "l'Industrie et du Numérique."


FR_MWT_TOKEN_TO_WORDS_GOLD = """
token: Alors    		words: [<Word id=1;text=Alors>]
token: encore   		words: [<Word id=2;text=encore>]
token: inconnu  		words: [<Word id=3;text=inconnu>]
token: du       		words: [<Word id=4;text=de>, <Word id=5;text=le>]
token: grand    		words: [<Word id=6;text=grand>]
token: public   		words: [<Word id=7;text=public>]
token: ,        		words: [<Word id=8;text=,>]
token: Emmanuel 		words: [<Word id=9;text=Emmanuel>]
token: Macron   		words: [<Word id=10;text=Macron>]
token: devient  		words: [<Word id=11;text=devient>]
token: en       		words: [<Word id=12;text=en>]
token: 2014     		words: [<Word id=13;text=2014>]
token: ministre 		words: [<Word id=14;text=ministre>]
token: de       		words: [<Word id=15;text=de>]
token: l'       		words: [<Word id=16;text=l'>]
token: Économie 		words: [<Word id=17;text=Économie>]
token: ,        		words: [<Word id=18;text=,>]
token: de       		words: [<Word id=19;text=de>]
token: l'       		words: [<Word id=20;text=l'>]
token: Industrie		words: [<Word id=21;text=Industrie>]
token: et       		words: [<Word id=22;text=et>]
token: du       		words: [<Word id=23;text=de>, <Word id=24;text=le>]
token: Numérique		words: [<Word id=25;text=Numérique>]
token: .        		words: [<Word id=26;text=.>]
""".strip()

FR_MWT_WORD_TO_TOKEN_GOLD = """
word: Alors    		token parent:1-Alors
word: encore   		token parent:2-encore
word: inconnu  		token parent:3-inconnu
word: de       		token parent:4-5-du
word: le       		token parent:4-5-du
word: grand    		token parent:6-grand
word: public   		token parent:7-public
word: ,        		token parent:8-,
word: Emmanuel 		token parent:9-Emmanuel
word: Macron   		token parent:10-Macron
word: devient  		token parent:11-devient
word: en       		token parent:12-en
word: 2014     		token parent:13-2014
word: ministre 		token parent:14-ministre
word: de       		token parent:15-de
word: l'       		token parent:16-l'
word: Économie 		token parent:17-Économie
word: ,        		token parent:18-,
word: de       		token parent:19-de
word: l'       		token parent:20-l'
word: Industrie		token parent:21-Industrie
word: et       		token parent:22-et
word: de       		token parent:23-24-du
word: le       		token parent:23-24-du
word: Numérique		token parent:25-Numérique
word: .        		token parent:26-.
""".strip()


def test_mwt():
    pipeline = stanza.Pipeline(processors='tokenize,mwt', dir=TEST_MODELS_DIR, lang='fr')
    doc = pipeline(FR_MWT_SENTENCE)
    token_to_words = "\n".join(
        [f'token: {token.text.ljust(9)}\t\twords: [{", ".join([word.pretty_print() for word in token.words])}]' for sent in doc.sentences for token in sent.tokens]
    ).strip()
    word_to_token = "\n".join(
        [f'word: {word.text.ljust(9)}\t\ttoken parent:{"-".join([str(x) for x in word.parent.id])}-{word.parent.text}'
         for sent in doc.sentences for word in sent.words]).strip()
    assert token_to_words == FR_MWT_TOKEN_TO_WORDS_GOLD
    assert word_to_token == FR_MWT_WORD_TO_TOKEN_GOLD
