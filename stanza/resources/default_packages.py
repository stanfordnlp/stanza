"""
Constants for default packages, default pretrains, charlms, etc

Separated from prepare_resources.py so that other modules can use the
same lists / maps without importing the resources script and possibly
causing a circular import
"""

import copy

# default treebank for languages
default_treebanks = {
    "af":      "afribooms",
    "grc":     "proiel",
    "ar":      "padt",
    "hy":      "armtdp",
    "eu":      "bdt",
    "bg":      "btb",
    "bxr":     "bdt",
    "ca":      "ancora",
    "zh-hant": "gsd",
    "hr":      "set",
    "cs":      "pdt",
    "cy":      "ccg",
    "da":      "ddt",
    "nl":      "alpino",
    "en":      "combined",
    "et":      "edt",
    "fi":      "tdt",
    "fo":      "farpahc",
    "fr":      "gsd",
    "gl":      "ctg",
    "de":      "gsd",
    "got":     "proiel",
    "el":      "gdt",
    "he":      "combined",
    "hi":      "hdtb",
    "hu":      "szeged",
    "id":      "gsd",
    "is":      "icepahc",
    "ga":      "idt",
    "it":      "combined",
    "ja":      "gsd",
    "kk":      "ktb",
    "ko":      "kaist",
    "kmr":     "mg",
    "la":      "ittb",
    "lij":     "glt",
    "lv":      "lvtb",
    "pcm":     "nsc",
    "sme":     "giella",
    "cu":      "proiel",
    "fro":     "srcmf",
    "fa":      "perdt",
    "my":      "ucsy",
    "myv":     "jr",
    "pl":      "pdb",
    "pt":      "bosque",
    "ro":      "rrt",
    "ru":      "syntagrus",
    "sa":      "vedic",
    "sd":      "isra",
    "sr":      "set",
    "sk":      "snk",
    "sl":      "ssj",
    "es":      "ancora",
    "sv":      "talbanken",
    "th":      "orchid",
    "tr":      "imst",
    "qtd":     "sagt",
    "uk":      "iu",
    "hsb":     "ufal",
    "ur":      "udtb",
    "ug":      "udt",
    "vi":      "vtb",
    "lt":      "alksnis",
    "hyw":     "armtdp",
    "wo":      "wtb",
    "nb":      "bokmaal",
    "mt":      "mudt",
    "swl":     "sslc",
    "cop":     "scriptorium",
    "be":      "hse",
    "zh-hans": "gsdsimp",
    "lzh":     "kyoto",
    "gd":      "arcosg",
    "olo":     "kkpp",
    "ta":      "ttb",
    "te":      "mtg",
    "orv":     "torot",
    "nn":      "nynorsk",
    "mr":      "ufal",
    "multilingual": "ud"
}

no_pretrain_languages = set([
    "cop",
    "orv",
    "pcm",
    "qtd",
    "swl",
])


# in some cases, we give the pretrain a name other than the original
# name for the UD dataset
# we will eventually do this for all of the pretrains
specific_default_pretrains = {
    "af":      "fasttextwiki",
    "ar":      "conll17",
    "be":      "fasttextwiki",
    "bg":      "conll17",
    "bxr":     "fasttextwiki",
    "ca":      "conll17",
    "cs":      "conll17",
    "cu":      "conll17",
    "cy":      "fasttext157",
    "da":      "conll17",
    "de":      "conll17",
    "el":      "conll17",
    "en":      "conll17",
    "es":      "conll17",
    "et":      "conll17",
    "eu":      "conll17",
    "fa":      "conll17",
    "fi":      "conll17",
    "fo":      "fasttextwiki",
    "fr":      "conll17",
    "fro":     "conll17",
    "ga":      "conll17",
    "gd":      "fasttextwiki",
    "gl":      "conll17",
    "got":     "fasttextwiki",
    "grc":     "conll17",
    "he":      "conll17",
    "hi":      "conll17",
    "hr":      "conll17",
    "hsb":     "fasttextwiki",
    "hu":      "conll17",
    "hy":      "isprasglove",
    "hyw":     "isprasglove",
    "id":      "conll17",
    "is":      "fasttext157",
    "it":      "conll17",
    "ja":      "conll17",
    "kk":      "fasttext157",
    "kmr":     "fasttextwiki",
    "ko":      "conll17",
    "la":      "conll17",
    "lij":     "fasttextwiki",
    "lt":      "fasttextwiki",
    "lv":      "conll17",
    "lzh":     "fasttextwiki",
    "mr":      "fasttextwiki",
    "mt":      "fasttextwiki",
    "myv":     "mokha",
    "nb":      "conll17",
    "nl":      "conll17",
    "nn":      "conll17",
    "pl":      "conll17",
    "pt":      "conll17",
    "ro":      "conll17",
    "ru":      "conll17",
    "sa":      "fasttext157",
    "sk":      "conll17",
    "sl":      "conll17",
    "sme":     "fasttextwiki",
    "sr":      "fasttextwiki",
    "sv":      "conll17",
    "ta":      "fasttextwiki",
    "te":      "fasttextwiki",
    "tr":      "conll17",
    "ug":      "conll17",
    "uk":      "conll17",
    "ur":      "conll17",
    "vi":      "conll17",
    "wo":      "fasttextwiki",
    "zh-hans": "fasttext157",
    "zh-hant": "conll17",
}

default_pretrains = dict(default_treebanks)
for lang in no_pretrain_languages:
    default_pretrains.pop(lang, None)
for lang in specific_default_pretrains.keys():
    default_pretrains[lang] = specific_default_pretrains[lang]


# TODO: eventually we want to
#   - rename all the pretrains to indicate where they are from
#   - only have special / unique names for the few which need it, such as the bio pretrains
pos_pretrains = {
    "en": {
        "combined_roberta": "combined",
        "combined_electra": "combined",
    },
}

depparse_pretrains = pos_pretrains

ner_pretrains = {
    "ar": {
        "aqmar": "fasttextwiki",
    },
    "de": {
        "conll03":      "fasttextwiki",
        # the bert version of germeval uses the smaller vector file
        "germeval2014": "fasttextwiki",
    },
    "en": {
        "anatem":       "craft",
        "bc4chemd":     "craft",
        "bc5cdr":       "craft",
        "bionlp13cg":   "craft",
        "jnlpba":       "craft",
        "linnaeus":     "craft",
        "ncbi_disease": "craft",
        "s800":         "craft",

        "ontonotes":    "fasttextcrawl",
        # the stanza-train sample NER model should use the default NER pretrain
        # for English, that is the same as ontonotes
        "sample":       "fasttextcrawl",

        "conll03":      "glove",

        "i2b2":         "mimic",
        "radiology":    "mimic",
    },
    "es": {
        "ancora":  "fasttextwiki",
        "conll02": "fasttextwiki",
    },
    "fr": {
        "wikiner": "fasttextwiki",
    },
    "nl": {
        "conll02": "fasttextwiki",
        "wikiner": "fasttextwiki",
    },
    "ru": {
        "wikiner": "fasttextwiki",
    },
    "th": {
        "lst20": "fasttext157",
    },
    "zh-hans": {
        "ontonotes": "fasttextwiki",
    }
}


# default charlms for languages
default_charlms = {
    "af": "oscar",
    "ar": "ccwiki",
    "bg": "conll17",
    "da": "oscar",
    "de": "newswiki",
    "en": "1billion",
    "es": "newswiki",
    "fa": "conll17",
    "fi": "conll17",
    "fr": "newswiki",
    "he": "oscar",
    "hi": "oscar",
    "id": "oscar2023",
    "it": "conll17",
    "ja": "conll17",
    "kk": "oscar",
    "mr": "l3cube",
    "my": "oscar",
    "nb": "conll17",
    "nl": "ccwiki",
    "pl": "oscar",
    "ru": "newswiki",
    "sd": "isra",
    "sv": "conll17",
    "te": "oscar2022",
    "th": "oscar",
    "tr": "conll17",
    "uk": "conll17",
    "vi": "conll17",
    "zh-hans": "gigaword"
}

pos_charlms = {
    "en": {
        # none of the English charlms help with craft or genia
        "craft": None,
        "genia": None,
        "mimic": "mimic",
    },
    "tr": {   # no idea why, but this particular one goes down in dev score
        "boun": None,
    },
}

depparse_charlms = copy.deepcopy(pos_charlms)

ner_charlms = {
    "en": {
        "conll03": "1billion",
        "ontonotes": "1billion",
        "anatem": "pubmed",
        "bc4chemd": "pubmed",
        "bc5cdr": "pubmed",
        "bionlp13cg": "pubmed",
        "i2b2": "mimic",
        "jnlpba": "pubmed",
        "linnaeus": "pubmed",
        "ncbi_disease": "pubmed",
        "radiology": "mimic",
        "s800": "pubmed",
    },
    "hu": {
        "combined": None,
    },
    "nn": {
        "norne": None,
    },
}

