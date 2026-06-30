"""
Global constants.

These language codes mirror UD language codes when possible
"""

from collections import defaultdict

import re

class UnknownLanguageError(ValueError):
    pass

# tuples in a list so we can assert that the langcodes are all unique
# When applicable, we favor the UD decision over any other possible
# language code or language name
# An example of this is sab -> Bokota, instead of bgd in ISO 693-3
# ISO 639-1 is out of date, but many of the UD datasets are labeled
# using the two letter abbreviations, so we add those for non-UD
# languages in the hopes that we've guessed right if those languages
# are eventually processed
# One source for the known languages UD may add in the future:
#   https://universaldependencies.org/languages.html
lcode2lang_raw = [
    ("abq", "Abaza"),
    ("ab",  "Abkhazian"),
    ("aa",  "Afar"),
    ("af",  "Afrikaans"),
    ("ak",  "Akan"),
    ("akk", "Akkadian"),
    ("aqz", "Akuntsu"),
    ("sq",  "Albanian"),
    ("am",  "Amharic"),
    ("grc", "Ancient_Greek"),
    ("hbo", "Ancient_Hebrew"),
    ("apy", "Apalai"),
    ("apu", "Apurina"),
    ("ar",  "Arabic"),
    ("an",  "Aragonese"),
    ("pgl", "Archaic_Irish"),
    ("hy",  "Armenian"),
    ("as",  "Assamese"),
    ("aii", "Assyrian"),
    ("ast", "Asturian"),
    ("av",  "Avaric"),
    ("ae",  "Avestan"),
    ("ay",  "Aymara"),
    ("az",  "Azerbaijani"),
    ("bav", "Babungo"),
    ("bal", "Balochi"),
    ("bm",  "Bambara"),
    ("ba",  "Bashkir"),
    ("eu",  "Basque"),
    ("ifb", "Batad_Ifugao"),
    ("bar", "Bavarian"),
    ("bej", "Beja"),
    ("be",  "Belarusian"),
    ("bn",  "Bengali"),
    ("bho", "Bhojpuri"),
    ("bpy", "Bishnupriya_Manipuri"),
    ("bi",  "Bislama"),
    ("sab", "Bokota"),
    ("peh", "Bonan"),
    ("bor", "Bororo"),
    ("bs",  "Bosnian"),
    ("brh", "Brahui"),
    ("br",  "Breton"),
    ("bzd", "Bribri"),
    ("bg",  "Bulgarian"),
    ("bxr", "Buryat"),
    ("yue", "Cantonese"),
    ("cpg", "Cappadocian"),
    ("ca",  "Catalan"),
    ("ceb", "Cebuano"),
    ("km",  "Central_Khmer"),
    ("rmc", "Central_Romani"),
    ("ch",  "Chamorro"),
    ("ce",  "Chechen"),
    ("ny",  "Chichewa"),
    ("ctn", "Chintang"),
    ("ckt", "Chukchi"),
    ("cv",  "Chuvash"),
    ("xcl", "Classical_Armenian"),
    ("lzh", "Classical_Chinese"),
    ("nci", "Classical_Nahuatl"),
    ("cop", "Coptic"),
    ("kw",  "Cornish"),
    ("co",  "Corsican"),
    ("cr",  "Cree"),
    ("hr",  "Croatian"),
    ("cux", "Cuicatec"),
    ("quz", "Cusco_Quechua"),
    ("cs",  "Czech"),
    ("da",  "Danish"),
    ("dar", "Dargwa"),
    ("prs", "Dari"),
    ("dta", "Daur"),
    ("dv",  "Dhivehi"),
    ("sce", "Dongxiang"),
    ("nl",  "Dutch"),
    ("dz",  "Dzongkha"),
    ("mhr", "Eastern_Mari"),
    ("egy", "Egyptian"),
    ("arz", "Egyptian_Arabic"),
    ("unk", "Enawene_Nawe"),
    ("en",  "English"),
    ("myv", "Erzya"),
    ("eo",  "Esperanto"),
    ("et",  "Estonian"),
    ("eve", "Even"),
    ("evn", "Evenki"),
    ("ee",  "Ewe"),
    ("ext", "Extremaduran"),
    ("fo",  "Faroese"),
    ("fj",  "Fijian"),
    ("fi",  "Finnish"),
    ("fon", "Fon"),
    ("fr",  "French"),
    ("fsl", "French_Sign_Language"),
    ("qfn", "Frisian_Dutch"),
    ("ff",  "Fulah"),
    ("gqa", "Ga"),
    ("gl",  "Galician"),
    ("lg",  "Ganda"),
    ("drs", "Gedeo"),
    ("ka",  "Georgian"),
    ("de",  "German"),
    ("aln", "Gheg"),
    ("bbj", "Ghomálá'"),
    ("glk", "Gilaki"),
    ("gom", "Goan_Konkani"),
    ("gor", "Gorontalo"),
    ("got", "Gothic"),
    ("el",  "Greek"),
    ("kl",  "Greenlandic"),
    ("gub", "Guajajara"),
    ("gn",  "Guarani"),
    ("gu",  "Gujarati"),
    ("gwi", "Gwichin"),
    ("ht",  "Haitian"),
    ("ha",  "Hausa"),
    ("he",  "Hebrew"),
    ("hz",  "Herero"),
    ("azz", "Highland_Puebla_Nahuatl"),
    ("hil", "Hiligaynon"),
    ("hi",  "Hindi"),
    ("qhe", "Hindi_English"),
    ("ho",  "Hiri_Motu"),
    ("hit", "Hittite"),
    ("huv", "Huave"),
    ("hu",  "Hungarian"),
    ("is",  "Icelandic"),
    ("io",  "Ido"),
    ("ig",  "Igbo"),
    ("arh", "Ika"),
    ("ilo", "Ilocano"),
    ("arc", "Imperial_Aramaic"),
    ("id",  "Indonesian"),
    ("ia",  "Interlingua"),
    ("ie",  "Interlingue"),
    ("iu",  "Inuktitut"),
    ("ik",  "Inupiaq"),
    ("ga",  "Irish"),
    ("it",  "Italian"),
    ("ja",  "Japanese"),
    ("jv",  "Javanese"),
    ("urb", "Kaapor"),
    ("kab", "Kabyle"),
    ("kbc", "Kadiweu"),
    ("xal", "Kalmyk"),
    ("xnr", "Kangri"),
    ("kn",  "Kannada"),
    ("kr",  "Kanuri"),
    ("pam", "Kapampangan"),
    ("krc", "Karachay_Balkar"),
    ("krl", "Karelian"),
    ("arr", "Karo"),
    ("ks",  "Kashmiri"),
    ("kk",  "Kazakh"),
    ("naq", "Khoekhoe"),
    ("kfm", "Khunsari"),
    ("quc", "Kiche"),
    ("cgg", "Kiga"),
    ("ki",  "Kikuyu"),
    ("rw",  "Kinyarwanda"),
    ("kv",  "Komi"),
    ("koi", "Komi_Permyak"),
    ("kpv", "Komi_Zyrian"),
    ("kg",  "Kongo"),
    ("ko",  "Korean"),
    ("kfz", "Koromfe"),
    ("ku",  "Kurdish"),
    ("kj",  "Kwanyama"),
    ("ky",  "Kyrgyz"),
    ("lad", "Ladino"),
    ("laj", "Lango"),
    ("lo",  "Lao"),
    ("ltg", "Latgalian"),
    ("la",  "Latin"),
    ("lv",  "Latvian"),
    ("lzz", "Laz"),
    ("lez", "Lezgian"),
    ("lij", "Ligurian"),
    ("li",  "Limburgish"),
    ("ln",  "Lingala"),
    ("lt",  "Lithuanian"),
    ("liv", "Livonian"),
    ("olo", "Livvi"),
    ("jbo", "Lojban"),
    ("lmo", "Lombard"),
    ("nds", "Low_Saxon"),
    ("dsb", "Lower_Sorbian"),
    ("lu",  "Luba_Katanga"),
    ("lb",  "Luxembourgish"),
    ("mk",  "Macedonian"),
    ("jaa", "Madi"),
    ("mag", "Magahi"),
    ("qaf", "Maghrebi_Arabic_French"),
    ("mai", "Maithili"),
    ("mpu", "Makurap"),
    ("mg",  "Malagasy"),
    ("ms",  "Malay"),
    ("ml",  "Malayalam"),
    ("mt",  "Maltese"),
    ("mnc", "Manchu"),
    ("mjl", "Mandyali"),
    ("mns", "Mansi"),
    ("gv",  "Manx"),
    ("mi",  "Maori"),
    ("mr",  "Marathi"),
    ("mh",  "Marshallese"),
    ("mxx", "Mauka"),
    ("mzn", "Mazandarani"),
    ("gun", "Mbya_Guarani"),
    ("axm", "Middle_Armenian"),
    ("enm", "Middle_English"),
    ("frm", "Middle_French"),
    ("mga", "Middle_Irish"),
    ("pal", "Middle_Persian"),
    ("min", "Minangkabau"),
    ("xmf", "Mingrelian"),
    ("mwl", "Mirandese"),
    ("lus", "Mizo"),
    ("mov", "Mojave"),
    ("mdf", "Moksha"),
    ("mn",  "Mongolian"),
    ("mos", "Mossi"),
    ("myu", "Munduruku"),
    ("zmu", "Muruwari"),
    ("my",  "Myanmar"),
    ("nqo", "N'Ko"),
    ("nmf", "Naga"),
    ("nah", "Nahuatl"),
    ("pcm", "Naija"),
    ("gld", "Nanai"),
    ("na",  "Nauru"),
    ("nv",  "Navajo"),
    ("nyq", "Nayini"),
    ("ndg", "Ndengeleko"),
    ("ng",  "Ndonga"),
    ("nap", "Neapolitan"),
    ("neg", "Negidal"),
    ("yrk", "Nenets"),
    ("ne",  "Nepali"),
    ("new", "Newar"),
    ("yrl", "Nheengatu"),
    ("nyn", "Nkore"),
    ("frr", "North_Frisian"),
    ("nd",  "North_Ndebele"),
    ("sme", "North_Sami"),
    ("hno", "Northern_Hindko"),
    ("kmr", "Northern_Kurdish"),
    ("lrc", "Northern_Luri"),
    ("nso", "Northern_Sotho"),
    ("gya", "Northwest_Gbaya"),
    ("nb",  "Norwegian_Bokmaal"),
    ("nn",  "Norwegian_Nynorsk"),
    ("ii",  "Nuosu"),
    ("oc",  "Occitan"),
    ("or",  "Odia"),
    ("oj",  "Ojibwa"),
    ("cu",  "Old_Church_Slavonic"),
    ("orv", "Old_East_Slavic"),
    ("ang", "Old_English"),
    ("fro", "Old_French"),
    ("oge", "Old_Georgian"),
    ("sga", "Old_Irish"),
    ("ojp", "Old_Japanese"),
    ("pro", "Old_Occitan"),
    ("osx", "Old_Saxon"),
    ("otk", "Old_Turkish"),
    ("oac", "Oroch"),
    ("om",  "Oromo"),
    ("os",  "Ossetian"),
    ("ota", "Ottoman_Turkish"),
    ("pln", "Palenquero"),
    ("pi",  "Pali"),
    ("pap", "Papiamento"),
    ("ps",  "Pashto"),
    ("pad", "Paumari"),
    ("mvf", "Peripheral_Mongolian"),
    ("fa",  "Persian"),
    ("pay", "Pesh"),
    ("xpg", "Phrygian"),
    ("pms", "Piedmontese"),
    ("pbv", "Pnar"),
    ("pl",  "Polish"),
    ("qpm", "Pomak"),
    ("pnt", "Pontic"),
    ("pt",  "Portuguese"),
    ("pra", "Prakrit"),
    ("pa",  "Punjabi"),
    ("qxp", "Puno_Quechua"),
    ("prx", "Purki"),
    ("qu",  "Quechua"),
    ("rhg", "Rohingya"),
    ("ro",  "Romanian"),
    ("rm",  "Romansh"),
    ("rn",  "Rundi"),
    ("ru",  "Russian"),
    ("rue", "Rusyn"),
    ("ruc", "Ruuli"),
    ("sm",  "Samoan"),
    ("sg",  "Sango"),
    ("sa",  "Sanskrit"),
    ("skr", "Saraiki"),
    ("sc",  "Sardinian"),
    ("sco", "Scots"),
    ("gd",  "Scottish_Gaelic"),
    ("sr",  "Serbian"),
    ("sei", "Seri"),
    ("wuu", "Shanghainese"),
    ("shp", "Shipibo_Konibo"),
    ("sn",  "Shona"),
    ("scn", "Sicilian"),
    ("zh-hans", "Simplified_Chinese"),
    ("sd",  "Sindhi"),
    ("si",  "Sinhala"),
    ("sms", "Skolt_Sami"),
    ("sk",  "Slovak"),
    ("sl",  "Slovenian"),
    ("soj", "Soi"),
    ("so",  "Somali"),
    ("ckb", "Sorani"),
    ("azb", "South_Azerbaijani"),
    ("ajp", "South_Levantine_Arabic"),
    ("nr",  "South_Ndebele"),
    ("hnd", "Southern_Hindko"),
    ("sdh", "Southern_Kurdish"),
    ("st",  "Southern_Sotho"),
    ("diq", "Southern_Zazaki"),
    ("es",  "Spanish"),
    ("ssp", "Spanish_Sign_Language"),
    ("su",  "Sundanese"),
    ("sw",  "Swahili"),
    ("ss",  "Swati"),
    ("sv",  "Swedish"),
    ("swl", "Swedish_Sign_Language"),
    ("gsw", "Swiss_German"),
    ("syr", "Syriac"),
    ("tl",  "Tagalog"),
    ("ty",  "Tahitian"),
    ("tg",  "Tajik"),
    ("ta",  "Tamil"),
    ("tt",  "Tatar"),
    ("eme", "Teko"),
    ("te",  "Telugu"),
    ("qte", "Telugu_English"),
    ("th",  "Thai"),
    ("bo",  "Tibetan"),
    ("ti",  "Tigrinya"),
    ("to",  "Tonga"),
    ("zh-hant", "Traditional_Chinese"),
    ("ts",  "Tsonga"),
    ("tn",  "Tswana"),
    ("tpn", "Tupinamba"),
    ("tr",  "Turkish"),
    ("qti", "Turkish_English"),
    ("qtd", "Turkish_German"),
    ("tk",  "Turkmen"),
    ("tyv", "Tuvinian"),
    ("tw",  "Twi"),
    ("uk",  "Ukrainian"),
    ("xum", "Umbrian"),
    ("hsb", "Upper_Sorbian"),
    ("ur",  "Urdu"),
    ("ug",  "Uyghur"),
    ("uz",  "Uzbek"),
    ("ve",  "Venda"),
    ("vep", "Veps"),
    ("vi",  "Vietnamese"),
    ("vo",  "Volapük"),
    ("wa",  "Walloon"),
    ("war", "Waray"),
    ("wbp", "Warlpiri"),
    ("cy",  "Welsh"),
    ("hyw", "Western_Armenian"),
    ("fy",  "Western_Frisian"),
    ("mrj", "Western_Mari"),
    ("pnb", "Western_Panjabi"),
    ("nhi", "Western_Sierra_Puebla_Nahuatl"),
    ("wo",  "Wolof"),
    ("xav", "Xavante"),
    ("xh",  "Xhosa"),
    ("sjo", "Xibe"),
    ("sah", "Yakut"),
    ("yi",  "Yiddish"),
    ("yo",  "Yoruba"),
    ("ess", "Yupik"),
    ("say", "Zaar"),
    ("zza", "Zazaki"),
    ("zea", "Zeelandic"),
    ("za",  "Zhuang"),
    ("zu",  "Zulu"),
]

# build the dictionary, checking for duplicate language codes
lcode2lang = {}
for code, language in lcode2lang_raw:
    assert code not in lcode2lang
    lcode2lang[code] = language

# invert the dictionary, checking for possible duplicate language names
lang2lcode = {}
for code, language in lcode2lang_raw:
    assert language not in lang2lcode
    lang2lcode[language] = code

# check that nothing got clobbered
assert len(lcode2lang_raw) == len(lcode2lang)
assert len(lcode2lang_raw) == len(lang2lcode)

# some of the two letter langcodes get used elsewhere as three letters
# for example, Wolof is abbreviated "wo" in UD, but "wol" in Masakhane NER
two_to_three_letters_raw = (
    ("bm",  "bam"),
    ("ee",  "ewe"),
    ("ha",  "hau"),
    ("ig",  "ibo"),
    ("rw",  "kin"),
    ("lg",  "lug"),
    ("ny",  "nya"),
    ("sn",  "sna"),
    ("sw",  "swa"),
    ("tn",  "tsn"),
    ("tw",  "twi"),
    ("wo",  "wol"),
    ("xh",  "xho"),
    ("yo",  "yor"),
    ("zu",  "zul"),

    # this is a weird case where a 2 letter code was available,
    # but UD used the 3 letter code instead
    ("se",  "sme"),
)

for two, three in two_to_three_letters_raw:
    if two in lcode2lang:
        assert two in lcode2lang
        assert three not in lcode2lang
        assert three not in lang2lcode
        lang2lcode[three] = two
        lcode2lang[three] = lcode2lang[two]
    elif three in lcode2lang:
        assert three in lcode2lang
        assert two not in lcode2lang
        assert two not in lang2lcode
        lang2lcode[two] = three
        lcode2lang[two] = lcode2lang[three]
    else:
        raise AssertionError("Found a proposed alias %s -> %s when neither code was already known" % (two, three))

two_to_three_letters = {
    two: three for two, three in two_to_three_letters_raw
}

three_to_two_letters = {
    three: two for two, three in two_to_three_letters_raw
}

assert len(two_to_three_letters) == len(two_to_three_letters_raw)
assert len(three_to_two_letters) == len(two_to_three_letters_raw)

# additional useful code to language mapping
# added after dict invert to avoid conflict
lcode2lang['bgd'] = 'Bokota'   # ISO 693-3 code, although UD used sab
lcode2lang['nb'] = 'Norwegian' # Norwegian Bokmall mapped to default norwegian
lcode2lang['no'] = 'Norwegian'
lcode2lang['zh'] = 'Simplified_Chinese'

# additional, less common names for languages already in lcode2lang
# this is a list of tuples rather than a dict, since a dict keyed by
# lcode would silently clobber any language that shares an lcode with
# another entry (eg. Divehi & Maldivian both -> dv).  each *language*
# name still needs to be unique, which is checked below
extra_lang_to_lcodes_raw = [
    ("Abkhaz", "ab"),
    ("Alemannic", "gsw"),
    ("Bangla", "bn"),
    ("Burmese", "my"),
    ("Central_Kurdish", "ckb"),
    ("Chewa", "ny"),
    ("Chinese", "zh"),
    ("Chuang", "za"),
    ("Divehi", "dv"),
    ("Emerillon", "eme"),
    ("Gaelic", "ga"),
    ("Genoese", "lij"),
    ("Gorkhali", "ne"),
    ("Haitian_Creole", "ht"),
    ("Iloko", "ilo"),
    ("Ilokano", "ilo"),
    ("isiNdebele", "nr"),
    ("isiXhosa", "xh"),
    ("isiZulu", "zu"),
    ("Jamamadí", "jaa"),
    ("Kabylian", "kab"),
    ("Kalaallisut", "kl"),
    ("Khmer", "km"),
    ("Kirghiz", "ky"),
    ("Kurmanji", "kmr"),
    ("Letzeburgesch", "lb"),
    ("Luganda", "lg"),
    ("Madí", "jaa"),
    ("Maldivian", "dv"),
    ("Mandeali", "mjl"),
    ("Multani", "skr"),
    ("Norwegian", "nb"),
    ("Nyanja", "ny"),
    ("Old_Gaelic", "sga"),
    # treebank names changed from Old Russian to Old East Slavic in 2.8
    ("Old_Russian", "orv"),
    ("Oriya", "or"),
    ("Ramarama", "arr"),
    ("Sakha", "sah"),
    ("Sepedi", "nso"),
    ("Sesotho", "st"),
    ("Setswana", "tn"),
    ("Sichuan_Yi", "ii"),
    ("Sinhalese", "si"),
    ("Siswati", "ss"),
    ("Sohi", "soj"),
    ("Tshivenda", "ve"),
    ("West_Frisian", "fy"),
    ("Wu_Chinese", "wuu"),
    ("Xitsonga", "ts"),
    ("Zaza", "zza"),
]

for language, code in extra_lang_to_lcodes_raw:
    assert language not in lang2lcode
    assert code in lcode2lang
    lang2lcode[language] = code

extra_lang_to_lcodes = {language: code for language, code in extra_lang_to_lcodes_raw}
assert len(extra_lang_to_lcodes) == len(extra_lang_to_lcodes_raw)

extra_lcode_to_lang = defaultdict(list)
for language, code in extra_lang_to_lcodes_raw:
    extra_lcode_to_lang[code].append(language)

# build a lowercase map from language to langcode
langlower2lcode = {}
for k in lang2lcode:
    langlower2lcode[k.lower()] = lang2lcode[k]

treebank_special_cases = {
    "UD_Chinese-Beginner": "zh-hans_beginner",
    "UD_Chinese-GSDSimp": "zh-hans_gsdsimp",
    "UD_Chinese-GSD": "zh-hant_gsd",
    "UD_Chinese-HK": "zh-hant_hk",
    "UD_Chinese-CFL": "zh-hans_cfl",
    "UD_Chinese-PatentChar": "zh-hans_patentchar",
    "UD_Chinese-PUD": "zh-hant_pud",
    "UD_Norwegian-Bokmaal": "nb_bokmaal",
    "UD_Norwegian-Nynorsk": "nn_nynorsk",
    "UD_Norwegian-NynorskLIA": "nn_nynorsklia",
}

SHORTNAME_RE = re.compile("^[a-z-]+_[a-z0-9-_]+$")

def langcode_to_lang(lcode):
    if lcode in lcode2lang:
        return lcode2lang[lcode]
    elif lcode.lower() in lcode2lang:
        return lcode2lang[lcode.lower()]
    else:
        return lcode

def pretty_langcode_to_lang(lcode):
    lang = langcode_to_lang(lcode)
    lang = lang.replace("_", " ")
    if lang == 'Simplified Chinese':
        lang = 'Chinese (Simplified)'
    elif lang == 'Traditional Chinese':
        lang = 'Chinese (Traditional)'
    return lang

def lang_to_langcode(lang):
    if lang in lang2lcode:
        lcode = lang2lcode[lang]
    elif lang.lower() in langlower2lcode:
        lcode = langlower2lcode[lang.lower()]
    elif lang in lcode2lang:
        lcode = lang
    elif lang.lower() in lcode2lang:
        lcode = lang.lower()
    else:
        raise UnknownLanguageError("Unable to find language code for %s" % lang)
    return lcode

# UG (Uyghur) has two commonly used scripts, but the UD dataset is the RtL script anyway
RIGHT_TO_LEFT = set(["aii", "ajb", "ar", "arc", "arz", "az", "azb", "bal", "ckb", "dv", "ff", "hbo", "he", "hnd", "hno", "ku", "lrc", "mzn", "nqo", "pnb", "prs", "ps", "fa", "rhg", "sd", "sdh", "skr", "syr", "ug", "ur", "yi"])

def is_right_to_left(lang):
    """
    Covers all the RtL languages we support, as well as many we don't.

    If a language is left out, please let us know!
    """
    lcode = lang_to_langcode(lang)
    return lcode in RIGHT_TO_LEFT

def treebank_to_short_name(treebank):
    """ Convert treebank name to short code. """
    if treebank in treebank_special_cases:
        return treebank_special_cases.get(treebank)
    if SHORTNAME_RE.match(treebank):
        lang, corpus = treebank.split("_", 1)
        lang = lang_to_langcode(lang)
        return lang + "_" + corpus

    if treebank.startswith('UD_'):
        treebank = treebank[3:]
    # special case starting with zh in case the input is an already-converted ZH treebank
    if treebank.startswith("zh-hans") or treebank.startswith("zh-hant"):
        splits = (treebank[:len("zh-hans")], treebank[len("zh-hans")+1:])
    elif treebank.endswith("-diacritics"):
        splits = treebank[:-11].split('-')
        splits[-1] = splits[-1] + "-diacritics"
    else:
        splits = treebank.split('-')
        if len(splits) == 1:
            splits = treebank.split("_", 1)
    assert len(splits) == 2, "Unable to process %s" % treebank
    lang, corpus = splits

    lcode = lang_to_langcode(lang)

    short = "{}_{}".format(lcode, corpus.lower())
    return short

def treebank_to_langid(treebank):
    """ Convert treebank name to langid """
    short_name = treebank_to_short_name(treebank)
    return short_name.split("_")[0]

