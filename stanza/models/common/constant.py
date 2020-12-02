"""
Global constants.

Please keep synced with
  scripts/treebank_to_shorthand.sh
"""

lcode2lang = {
    "af": "Afrikaans",
    "akk": "Akkadian",
    "aqz": "Akuntsu",
    "sq": "Albanian",
    "am": "Amharic",
    "grc": "Ancient_Greek",
    "apu": "Apurina",
    "ar": "Arabic",
    "hy": "Armenian",
    "aii": "Assyrian",
    "bm": "Bambara",
    "eu": "Basque",
    "be": "Belarusian",
    "bho": "Bhojpuri",
    "br": "Breton",
    "bg": "Bulgarian",
    "bxr": "Buryat",
    "yue": "Cantonese",
    "ca": "Catalan",
    "zh-hant": "Traditional_Chinese",
    "lzh": "Classical_Chinese",
    "ckt": "Chukchi",
    "cop": "Coptic",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "myv": "Erzya",
    "et": "Estonian",
    "fo": "Faroese",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "got": "Gothic",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "qhe": "Hindi_English",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "krl": "Karelian",
    "kk": "Kazakh",
    "kfm": "Khunsari",
    "koi": "Komi_Permyak",
    "kpv": "Komi_Zyrian",
    "ko": "Korean",
    "kmr": "Kurmanji",
    "lt": "Lithuanian",
    "olo": "Livvi",
    "la": "Latin",
    "lv": "Latvian",
    "mt": "Maltese",
    "gv": "Manx",
    "mr": "Marathi",
    "gun": "Mbya_Guarani",
    "mdf": "Moksha",
    "myu": "Munduruku",
    "pcm": "Naija",
    "nyq": "Nayini",
    "sme": "North_Sami",
    "nb": "Norwegian_Bokmaal",
    "nn": "Norwegian_Nynorsk",
    "cu": "Old_Church_Slavonic",
    "fro": "Old_French",
    "orv": "Old_Russian",
    "otk": "Old_Turkish",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "gd": "Scottish_Gaelic",
    "sr": "Serbian",
    "zh-hans": "Simplified_Chinese",
    "sms": "Skolt_Sami",
    "sk": "Slovak",
    "sl": "Slovenian",
    "soi": "Soi",
    "ajp": "South_Levantine_Arabic",
    "es": "Spanish",
    "sv": "Swedish",
    "swl": "Swedish_Sign_Language",
    "gsw": "Swiss_German",
    "tl": "Tagalog",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tpn": "Tupinamba",
    "tr": "Turkish",
    "qtd": "Turkish_German",
    "uk": "Ukrainian",
    "hsb": "Upper_Sorbian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "vi": "Vietnamese",
    "wbp": "Warlpipi",
    "cy": "Welsh",
    "wo": "Wolof",
    "yo": "Yoruba",
}

lang2lcode = {lcode2lang[k]: k for k in lcode2lang}
langlower2lcode = {lcode2lang[k].lower(): k.lower() for k in lcode2lang}

# additional useful code to language mapping
# added after dict invert to avoid conflict
lcode2lang['nb'] = 'Norwegian' # Norwegian Bokmall mapped to default norwegian
lcode2lang['zh'] = 'Simplified_Chinese'

lang2lcode['Chinese'] = 'zh'

treebank_special_cases = {
    "UD_Chinese-GSDSimp": "zh_gsdsimp",
    "UD_Chinese-GSD": "zh-hant_gsd",
    "UD_Chinese-HK": "zh-hant_hk",
    "UD_Chinese-CFL": "zh-hant_cfl",
    "UD_Chinese-PUD": "zh-hant_pud",
    "UD_Norwegian-Bokmaal": "nb_bokmaal",
    "UD_Norwegian-Nynorsk": "nn_nynorsk",
    "UD_Norwegian-NynorskLIA": "nn_nynorsklia",
}

def treebank_to_short_name(treebank):
    """ Convert treebank name to short code. """
    if treebank in treebank_special_cases:
        return treebank_special_cases.get(treebank)

    if treebank.startswith('UD_'):
        treebank = treebank[3:]
    splits = treebank.split('-')
    assert len(splits) == 2
    lang, corpus = splits

    lcode = lang2lcode[lang]

    short = "{}_{}".format(lcode, corpus.lower())
    return short

