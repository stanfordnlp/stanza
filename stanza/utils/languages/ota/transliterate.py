import re

def ota_converter(a):
    a = re.sub("(ˇ|˘|ʷ)", '', a)
    a = re.sub('À', 'â', a)
    a = re.sub('æ', 's', a)
    a = re.sub('İ', 'i', a)
    a = re.sub('Ī', 'î', a)
    a = re.sub('Ç', 'ç', a)
    a = re.sub('Ş', 'ş', a)
    a = a.lower()
    a = re.sub('¶i', "", a)
    a = re.sub("‘|'|è|ê|ʿ|῾|ˀ|᾽", "’", a)
    a = re.sub('(ḍ|ē)', 'd', a)
    a = re.sub('ġ', 'g', a)
    a = re.sub('(ḳ̇̄|ḳ|ķ|ú)', 'k', a)
    a = re.sub('(ẖ|ģ|ħ|ĥ|ḥ|Ò|ò|Ò|ḫ|ó)', 'h', a)
    a = re.sub('(ā|à|À)', 'â', a)
    a = re.sub('(ì|ì|ī|ì)', 'î', a)
    a = re.sub("ō", "ô", a)
    a = re.sub('(s̱|ŝ|ś|ṯ|å|ã|ä|ṣ)', 's', a)
    a = re.sub('(ù|ẅ|ṭ|š)', 't', a)
    a = re.sub('(ū|ÿ)', 'û', a)
    a = re.sub('̇̄v', 'v', a)
    a = re.sub('(ẓ|ź|ø|ż|ẕ|ž|õ)', 'z', a)
    a = re.sub('(ė̄|ė)', 'e', a)
    # The correct letter should be ñ but modern Turkish doesn't have this.
    a = re.sub('(ñ̄|ň|ŋ)', 'n', a) 

    return a
