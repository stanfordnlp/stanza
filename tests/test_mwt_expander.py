"""
Basic testing of multi-word-token expansion
"""

import stanfordnlp

from tests import *


# mwt data for testing
FR_MWT_SENTENCE = "Alors encore inconnu du grand public, Emmanuel Macron devient en 2014 ministre de l'Économie, de " \
                  "l'Industrie et du Numérique."


FR_MWT_TOKEN_TO_WORDS_GOLD = """
token: Alors    		words: [<Word index=1;text=Alors>]
token: encore   		words: [<Word index=2;text=encore>]
token: inconnu  		words: [<Word index=3;text=inconnu>]
token: du       		words: [<Word index=4;text=de>, <Word index=5;text=le>]
token: grand    		words: [<Word index=6;text=grand>]
token: public   		words: [<Word index=7;text=public>]
token: ,        		words: [<Word index=8;text=,>]
token: Emmanuel 		words: [<Word index=9;text=Emmanuel>]
token: Macron   		words: [<Word index=10;text=Macron>]
token: devient  		words: [<Word index=11;text=devient>]
token: en       		words: [<Word index=12;text=en>]
token: 2014     		words: [<Word index=13;text=2014>]
token: ministre 		words: [<Word index=14;text=ministre>]
token: de       		words: [<Word index=15;text=de>]
token: l'       		words: [<Word index=16;text=l'>]
token: Économie 		words: [<Word index=17;text=Économie>]
token: ,        		words: [<Word index=18;text=,>]
token: de       		words: [<Word index=19;text=de>]
token: l'       		words: [<Word index=20;text=l'>]
token: Industrie		words: [<Word index=21;text=Industrie>]
token: et       		words: [<Word index=22;text=et>]
token: du       		words: [<Word index=23;text=de>, <Word index=24;text=le>]
token: Numérique		words: [<Word index=25;text=Numérique>]
token: .        		words: [<Word index=26;text=.>]
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
    pipeline = stanfordnlp.Pipeline(processors='tokenize,mwt', models_dir=TEST_MODELS_DIR, lang='fr')
    doc = pipeline(FR_MWT_SENTENCE)
    token_to_words = "\n".join(
        [f'token: {token.text.ljust(9)}\t\twords: {token.words}' for sent in doc.sentences for token in sent.tokens]
    ).strip()
    word_to_token = "\n".join(
        [f'word: {word.text.ljust(9)}\t\ttoken parent:{word.parent_token.index+"-"+word.parent_token.text}'
         for sent in doc.sentences for word in sent.words]).strip()
    assert token_to_words == FR_MWT_TOKEN_TO_WORDS_GOLD
    assert word_to_token == FR_MWT_WORD_TO_TOKEN_GOLD
