"""
Convert an input language name into its short code.
"""
import sys

from stanza.models.common.constant import lang2lcode

if len(sys.argv) <= 1:
    raise Exception("Language name not provided.")

lang = sys.argv[1]
if lang not in lang2lcode:
    raise Exception("Language name not found: {}".format(lang))
code = lang2lcode[lang]
sys.stdout.write(code)

