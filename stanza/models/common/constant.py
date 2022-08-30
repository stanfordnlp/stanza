"""
Global constants.

These language codes mirror UD language codes when possible
"""

import re

# tuples in a list so we can assert that the langcodes are all unique
# When applicable, we favor the UD decision over any other possible
# language code or language name
# ISO 639-1 is out of date, but many of the UD datasets are labeled
# using the two letter abbreviations, so we add those for non-UD
# languages in the hopes that we've guessed right if those languages
# are eventually processed
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
    ("apu", "Apurina"),
    ("ar",  "Arabic"),
    ("an",  "Aragonese"),
    ("hy",  "Armenian"),
    ("as",  "Assamese"),
    ("aii", "Assyrian"),
    ("ast", "Asturian"),
    ("av",  "Avaric"),
    ("ae",  "Avestan"),
    ("ay",  "Aymara"),
    ("az",  "Azerbaijani"),
    ("bm",  "Bambara"),
    ("ba",  "Bashkir"),
    ("eu",  "Basque"),
    ("bar", "Bavarian"),
    ("bej", "Beja"),
    ("be",  "Belarusian"),
    ("bn",  "Bengali"),
    ("bho", "Bhojpuri"),
    ("bpy", "Bishnupriya_Manipuri"),
    ("bi",  "Bislama"),
    ("bs",  "Bosnian"),
    ("br",  "Breton"),
    ("bg",  "Bulgarian"),
    ("bxr", "Buryat"),
    ("yue", "Cantonese"),
    ("cpg", "Cappadocian"),
    ("ca",  "Catalan"),
    ("ceb", "Cebuano"),
    ("km",  "Central_Khmer"),
    ("ch",  "Chamorro"),
    ("ce",  "Chechen"),
    ("ny",  "Chichewa"),
    ("ckt", "Chukchi"),
    ("cv",  "Chuvash"),
    ("lzh", "Classical_Chinese"),
    ("cop", "Coptic"),
    ("kw",  "Cornish"),
    ("co",  "Corsican"),
    ("cr",  "Cree"),
    ("hr",  "Croatian"),
    ("cs",  "Czech"),
    ("da",  "Danish"),
    ("dar", "Dargwa"),
    ("dv",  "Dhivehi"),
    ("nl",  "Dutch"),
    ("dz",  "Dzongkha"),
    ("en",  "English"),
    ("myv", "Erzya"),
    ("eo",  "Esperanto"),
    ("et",  "Estonian"),
    ("ee",  "Ewe"),
    ("fo",  "Faroese"),
    ("fj",  "Fijian"),
    ("fi",  "Finnish"),
    ("fon", "Fon"),
    ("fr",  "French"),
    ("qfn", "Frisian_Dutch"),
    ("ff",  "Fulah"),
    ("gl",  "Galician"),
    ("lg",  "Ganda"),
    ("ka",  "Georgian"),
    ("de",  "German"),
    ("bbj", "Ghomálá'"),
    ("got", "Gothic"),
    ("el",  "Greek"),
    ("kl",  "Greenlandic"),
    ("gub", "Guajajara"),
    ("gn",  "Guarani"),
    ("gu",  "Gujarati"),
    ("ht",  "Haitian"),
    ("ha",  "Hausa"),
    ("he",  "Hebrew"),
    ("hz",  "Herero"),
    ("hil", "Hiligaynon"),
    ("hi",  "Hindi"),
    ("qhe", "Hindi_English"),
    ("ho",  "Hiri_Motu"),
    ("hit", "Hittite"),
    ("hu",  "Hungarian"),
    ("is",  "Icelandic"),
    ("io",  "Ido"),
    ("ig",  "Igbo"),
    ("ilo", "Ilocano"),
    ("arc", "Imperial_Aramaic"),
    ("id",  "Indonesian"),
    ("iu",  "Inuktitut"),
    ("ik",  "Inupiaq"),
    ("ga",  "Irish"),
    ("it",  "Italian"),
    ("ja",  "Japanese"),
    ("jv",  "Javanese"),
    ("urb", "Kaapor"),
    ("kab", "Kabyle"),
    ("xnr", "Kangri"),
    ("kn",  "Kannada"),
    ("kr",  "Kanuri"),
    ("pam", "Kapampangan"),
    ("krl", "Karelian"),
    ("arr", "Karo"),
    ("ks",  "Kashmiri"),
    ("kk",  "Kazakh"),
    ("kfm", "Khunsari"),
    ("quc", "Kiche"),
    ("cgg", "Kiga"),
    ("ki",  "Kikuyu"),
    ("rw",  "Kinyarwanda"),
    ("ky",  "Kyrgyz"),
    ("kv",  "Komi"),
    ("koi", "Komi_Permyak"),
    ("kpv", "Komi_Zyrian"),
    ("kg",  "Kongo"),
    ("ko",  "Korean"),
    ("ku",  "Kurdish"),
    ("kmr", "Kurmanji"),
    ("kj",  "Kwanyama"),
    ("lad", "Ladino"),
    ("lo",  "Lao"),
    ("la",  "Latin"),
    ("lv",  "Latvian"),
    ("lij", "Ligurian"),
    ("li",  "Limburgish"),
    ("ln",  "Lingala"),
    ("lt",  "Lithuanian"),
    ("olo", "Livvi"),
    ("nds", "Low_Saxon"),
    ("lu",  "Luba_Katanga"),
    ("lb",  "Luxembourgish"),
    ("mk",  "Macedonian"),
    ("jaa", "Madi"),
    ("mag", "Magahi"),
    ("mai", "Maithili"),
    ("mpu", "Makurap"),
    ("mg",  "Malagasy"),
    ("ms",  "Malay"),
    ("ml",  "Malayalam"),
    ("mt",  "Maltese"),
    ("mjl", "Mandyali"),
    ("gv",  "Manx"),
    ("mi",  "Maori"),
    ("mr",  "Marathi"),
    ("mh",  "Marshallese"),
    ("mzn", "Mazandarani"),
    ("gun", "Mbya_Guarani"),
    ("enm", "Middle_English"),
    ("min", "Minangkabau"),
    ("xmf", "Mingrelian"),
    ("mwl", "Mirandese"),
    ("mdf", "Moksha"),
    ("mn",  "Mongolian"),
    ("mos", "Mossi"),
    ("myu", "Munduruku"),
    ("my",  "Myanmar"),
    ("nqo", "N'Ko"),
    ("nah", "Nahuatl"),
    ("pcm", "Naija"),
    ("na",  "Nauru"),
    ("nv",  "Navajo"),
    ("nyq", "Nayini"),
    ("ng",  "Ndonga"),
    ("nap", "Neapolitan"),
    ("ne",  "Nepali"),
    ("new", "Newar"),
    ("nyn", "Nkore"),
    ("frr", "North_Frisian"),
    ("nd",  "North_Ndebele"),
    ("sme", "North_Sami"),
    ("se",  "Northern_Sami"),
    ("nso", "Northern_Sotho"),
    ("nb",  "Norwegian_Bokmaal"),
    ("nn",  "Norwegian_Nynorsk"),
    ("ii",  "Nuosu"),
    ("oc",  "Occitan"),
    ("oj",  "Ojibwa"),
    ("cu",  "Old_Church_Slavonic"),
    ("orv", "Old_East_Slavic"),
    ("ang", "Old_English"),
    ("fro", "Old_French"),
    ("sga", "Old_Irish"),
    ("ojp", "Old_Japanese"),
    ("otk", "Old_Turkish"),
    ("or",  "Odia"),
    ("om",  "Oromo"),
    ("os",  "Ossetian"),
    ("pi",  "Pali"),
    ("ps",  "Pashto"),
    ("fa",  "Persian"),
    ("pbv", "Pnar"),
    ("pl",  "Polish"),
    ("qpm", "Pomak"),
    ("pnt", "Pontic"),
    ("pt",  "Portuguese"),
    ("pra", "Prakrit"),
    ("pa",  "Punjabi"),
    ("qu",  "Quechua"),
    ("rhg", "Rohingya"),
    ("ro",  "Romanian"),
    ("rm",  "Romansh"),
    ("rn",  "Rundi"),
    ("ru",  "Russian"),
    ("sm",  "Samoan"),
    ("sg",  "Sango"),
    ("sa",  "Sanskrit"),
    ("skr", "Saraiki"),
    ("sc",  "Sardinian"),
    ("sco", "Scots"),
    ("gd",  "Scottish_Gaelic"),
    ("sr",  "Serbian"),
    ("sn",  "Shona"),
    ("zh-hans", "Simplified_Chinese"),
    ("sd",  "Sindhi"),
    ("si",  "Sinhala"),
    ("sms", "Skolt_Sami"),
    ("sk",  "Slovak"),
    ("sl",  "Slovenian"),
    ("soj", "Soi"),
    ("so",  "Somali"),
    ("ckb", "Sorani"),
    ("ajp", "South_Levantine_Arabic"),
    ("nr",  "South_Ndebele"),
    ("st",  "Southern_Sotho"),
    ("es",  "Spanish"),
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
    ("th",  "Thai"),
    ("bo",  "Tibetan"),
    ("ti",  "Tigrinya"),
    ("to",  "Tonga"),
    ("zh-hant", "Traditional_Chinese"),
    ("ts",  "Tsonga"),
    ("tn",  "Tswana"),
    ("tpn", "Tupinamba"),
    ("tr",  "Turkish"),
    ("qtd", "Turkish_German"),
    ("tk",  "Turkmen"),
    ("tw",  "Twi"),
    ("uk",  "Ukrainian"),
    ("xum", "Umbrian"),
    ("hsb", "Upper_Sorbian"),
    ("ur",  "Urdu"),
    ("ug",  "Uyghur"),
    ("uz",  "Uzbek"),
    ("ve",  "Venda"),
    ("vi",  "Vietnamese"),
    ("vo",  "Volapük"),
    ("wa",  "Walloon"),
    ("war", "Waray"),
    ("wbp", "Warlpiri"),
    ("cy",  "Welsh"),
    ("hyw", "Western_Armenian"),
    ("fy",  "Western_Frisian"),
    ("wo",  "Wolof"),
    ("xh",  "Xhosa"),
    ("sjo", "Xibe"),
    ("sah", "Yakut"),
    ("yi",  "Yiddish"),
    ("yo",  "Yoruba"),
    ("ess", "Yupik"),
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
)

for two, three in two_to_three_letters_raw:
    assert two in lcode2lang
    assert three not in lcode2lang
    assert three not in lang2lcode
    lang2lcode[three] = two
    lcode2lang[three] = lcode2lang[two]

two_to_three_letters = {
    two: three for two, three in two_to_three_letters_raw
}

assert len(two_to_three_letters) == len(two_to_three_letters_raw)

# additional useful code to language mapping
# added after dict invert to avoid conflict
lcode2lang['nb'] = 'Norwegian' # Norwegian Bokmall mapped to default norwegian
lcode2lang['no'] = 'Norwegian'
lcode2lang['zh'] = 'Simplified_Chinese'

extra_lang_to_lcodes = [
    ("gsw", "Alemannic"),
    ("my",  "Burmese"),
    ("ckb", "Central_Kurdish"),
    ("ny",  "Chewa"),
    ("zh",  "Chinese"),
    ("za",  "Chuang"),
    ("dv",  "Divehi"),
    ("eme", "Emerillon"),
    ("lij", "Genoese"),
    ("ga",  "Gaelic"),
    ("ne",  "Gorkhali"),
    ("ilo", "Ilokano"),
    ("nr",  "isiNdebele"),
    ("xh",  "isiXhosa"),
    ("zu",  "isiZulu"),
    ("jaa", "Jamamadí"),
    ("kab", "Kabylian"),
    ("kl",  "Kalaallisut"),
    ("km",  "Khmer"),
    ("ky",  "Kirghiz"),
    ("lb",  "Letzeburgesch"),
    ("lg",  "Luganda"),
    ("jaa", "Madí"),
    ("dv",  "Maldivian"),
    ("mjl", "Mandeali"),
    ("skr", "Multani"),
    ("nb",  "Norwegian"),
    ("ny",  "Nyanja"),
    ("sga", "Old_Gaelic"),
    ("or",  "Oriya"),
    ("arr", "Ramarama"),
    ("sah", "Sakha"),
    ("nso", "Sepedi"),
    ("tn",  "Setswana"),
    ("ii",  "Sichuan_Yi"),
    ("si",  "Sinhalese"),
    ("ss",  "Siswati"),
    ("soj", "Sohi"),
    ("st",  "Sesotho"),
    ("ve",  "Tshivenda"),
    ("ts",  "Xitsonga"),
    ("fy",  "West_Frisian"),
    ("zza", "Zaza"),
]

for code, language in extra_lang_to_lcodes:
    assert language not in lang2lcode
    assert code in lcode2lang
    lang2lcode[language] = code

# treebank names changed from Old Russian to Old East Slavic in 2.8
lang2lcode['Old_Russian'] = 'orv'

# build a lowercase map from language to langcode
langlower2lcode = {}
for k in lang2lcode:
    langlower2lcode[k.lower()] = lang2lcode[k]

treebank_special_cases = {
    "UD_Chinese-GSDSimp": "zh-hans_gsdsimp",
    "UD_Chinese-GSD": "zh-hant_gsd",
    "UD_Chinese-HK": "zh-hant_hk",
    "UD_Chinese-CFL": "zh-hans_cfl",
    "UD_Chinese-PUD": "zh-hant_pud",
    "UD_Norwegian-Bokmaal": "no_bokmaal",
    "UD_Norwegian-Nynorsk": "nn_nynorsk",
    "UD_Norwegian-NynorskLIA": "nn_nynorsklia",
}

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")

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
        raise ValueError("Unable to find language code for %s" % lang)
    return lcode

RIGHT_TO_LEFT = set(["ar", "arc", "az", "ckb", "dv", "ff", "he", "ku", "mzn", "nqo", "ps", "fa", "rhg", "sd", "syr", "ur"])

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
        lang, corpus = treebank.split("_")
        lang = lang_to_langcode(lang)
        return lang + "_" + corpus

    if treebank.startswith('UD_'):
        treebank = treebank[3:]
    # special case starting with zh in case the input is an already-converted ZH treebank
    if treebank.startswith("zh-hans") or treebank.startswith("zh-hant"):
        splits = (treebank[:len("zh-hans")], treebank[len("zh-hans")+1:])
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

