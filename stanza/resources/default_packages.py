"""
Constants for default packages, default pretrains, charlms, etc

Separated from prepare_resources.py so that other modules can use the
same lists / maps without importing the resources script and possibly
causing a circular import
"""

import copy

# all languages will have a map which represents the available packages
PACKAGES = "packages"

# default treebank for languages
default_treebanks = {
    "af":      "afribooms",
    # currently not publicly released!  sent to us from the group developing this resource
    "ang":     "nerthus",
    "ar":      "padt",
    "be":      "hse",
    "bg":      "btb",
    "bxr":     "bdt",
    "ca":      "ancora",
    "cop":     "scriptorium",
    "cs":      "pdt",
    "cu":      "proiel",
    "cy":      "ccg",
    "da":      "ddt",
    "de":      "gsd",
    "el":      "gdt",
    "en":      "combined",
    "es":      "combined",
    "et":      "edt",
    "eu":      "bdt",
    "fa":      "perdt",
    "fi":      "tdt",
    "fo":      "farpahc",
    "fr":      "combined",
    "fro":     "profiterole",
    "ga":      "idt",
    "gd":      "arcosg",
    "gl":      "ctg",
    "got":     "proiel",
    "grc":     "perseus",
    "gv":      "cadhan",
    "hbo":     "ptnk",
    "he":      "combined",
    "hi":      "hdtb",
    "hr":      "set",
    "hsb":     "ufal",
    "hu":      "szeged",
    "hy":      "armtdp",
    "hyw":     "armtdp",
    "id":      "gsd",
    "is":      "icepahc",
    "it":      "combined",
    "ja":      "gsd",
    "ka":      "glc",
    "kk":      "ktb",
    "kmr":     "mg",
    "ko":      "kaist",
    "kpv":     "lattice",
    "ky":      "ktmu",
    "la":      "ittb",
    "lij":     "glt",
    "lt":      "alksnis",
    "lv":      "lvtb",
    "lzh":     "kyoto",
    "mr":      "ufal",
    "mt":      "mudt",
    "my":      "ucsy",
    "myv":     "jr",
    "nb":      "bokmaal",
    "nds":     "lsdc",
    "nl":      "alpino",
    "nn":      "nynorsk",
    "olo":     "kkpp",
    "orv":     "torot",
    "ota":     "boun",
    "pcm":     "nsc",
    "pl":      "pdb",
    "pt":      "bosque",
    "qaf":     "arabizi",
    "qpm":     "philotis",
    "qtd":     "sagt",
    "ro":      "rrt",
    "ru":      "syntagrus",
    "sa":      "vedic",
    "sd":      "isra",
    "sk":      "snk",
    "sl":      "ssj",
    "sme":     "giella",
    "sq":      "combined",
    "sr":      "set",
    "sv":      "talbanken",
    "swl":     "sslc",
    "ta":      "ttb",
    "te":      "mtg",
    "th":      "orchid",
    "tr":      "imst",
    "ug":      "udt",
    "uk":      "iu",
    "ur":      "udtb",
    "vi":      "vtb",
    "wo":      "wtb",
    "xcl":     "caval",
    "zh-hans": "gsdsimp",
    "zh-hant": "gsd",
    "multilingual": "ud"
}

no_pretrain_languages = set([
    "cop",
    "olo",
    "orv",
    "pcm",
    "qaf",   # the QAF treebank is code switched and Romanized, so not easy to reuse existing resources
    "qpm",   # have talked about deriving this from a language neighborinig to Pomak, but that hasn't happened yet
    "qtd",
    "swl",

    "multilingual", # special case so that all languages with a default treebank are represented somewhere
])


# in some cases, we give the pretrain a name other than the original
# name for the UD dataset
# we will eventually do this for all of the pretrains
specific_default_pretrains = {
    "af":      "fasttextwiki",
    "ang":     "nerthus",
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
    "gv":      "fasttext157",
    "hbo":     "utah",
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
    "ka":      "fasttext157",
    "kk":      "fasttext157",
    "kmr":     "fasttextwiki",
    "ko":      "conll17",
    "kpv":     "fasttextwiki",
    "ky":      "fasttext157",
    "la":      "conll17",
    "lij":     "fasttextwiki",
    "lt":      "fasttextwiki",
    "lv":      "conll17",
    "lzh":     "fasttextwiki",
    "mr":      "fasttextwiki",
    "mt":      "fasttextwiki",
    "my":      "ucsy",
    "myv":     "mokha",
    "nb":      "conll17",
    "nds":     "fasttext157",
    "nl":      "conll17",
    "nn":      "conll17",
    "ota":     "conll17",
    "pl":      "conll17",
    "pt":      "conll17",
    "ro":      "conll17",
    "ru":      "conll17",
    "sa":      "fasttext157",
    "sd":      "isra",
    "sk":      "conll17",
    "sl":      "conll17",
    "sme":     "fasttextwiki",
    "sq":      "fasttext157",
    "sr":      "fasttextwiki",
    "sv":      "conll17",
    "ta":      "fasttextwiki",
    "te":      "fasttextwiki",
    "th":      "fasttext157",
    "tr":      "conll17",
    "ug":      "conll17",
    "uk":      "conll17",
    "ur":      "conll17",
    "vi":      "conll17",
    "wo":      "fasttextwiki",
    "xcl":     "caval",
    "zh-hans": "fasttext157",
    "zh-hant": "conll17",
}


def build_default_pretrains(default_treebanks):
    default_pretrains = dict(default_treebanks)
    for lang in no_pretrain_languages:
        default_pretrains.pop(lang, None)
    for lang in specific_default_pretrains.keys():
        default_pretrains[lang] = specific_default_pretrains[lang]
    return default_pretrains

default_pretrains = build_default_pretrains(default_treebanks)

pos_pretrains = {
    "en": {
        "craft":            "biomed",
        "genia":            "biomed",
        "mimic":            "mimic",
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
        "anatem":       "biomed",
        "bc4chemd":     "biomed",
        "bc5cdr":       "biomed",
        "bionlp13cg":   "biomed",
        "jnlpba":       "biomed",
        "linnaeus":     "biomed",
        "ncbi_disease": "biomed",
        "s800":         "biomed",

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
}


# default charlms for languages
default_charlms = {
    "af": "oscar",
    "ang": "nerthus1024",
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
    "pt": "oscar2023",
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

lemma_charlms = copy.deepcopy(pos_charlms)

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

# default ner for languages
default_ners = {
    "af": "nchlt",
    "ar": "aqmar_charlm",
    "bg": "bsnlp19",
    "da": "ddt",
    "de": "germeval2014",
    "en": "ontonotes-ww-multi_charlm",
    "es": "conll02",
    "fa": "arman",
    "fi": "turku",
    "fr": "wikinergold_charlm",
    "he": "iahlt_charlm",
    "hu": "combined",
    "hy": "armtdp",
    "it": "fbk",
    "ja": "gsd",
    "kk": "kazNERD",
    "mr": "l3cube",
    "my": "ucsy",
    "nb": "norne",
    "nl": "conll02",
    "nn": "norne",
    "pl": "nkjp",
    "ru": "wikiner",
    "sd": "siner",
    "sv": "suc3shuffle",
    "th": "lst20",
    "tr": "starlang",
    "uk": "languk",
    "vi": "vlsp",
    "zh-hans": "ontonotes",
}

# a few languages have sentiment classifier models
default_sentiment = {
    "en": "sstplus_charlm",
    "de": "sb10k_charlm",
    "es": "tass2020_charlm",
    "mr": "l3cube_charlm",
    "vi": "vsfc_charlm",
    "zh-hans": "ren_charlm",
}

# also, a few languages (very few, currently) have constituency parser models
default_constituency = {
    "da": "arboretum_charlm",
    "de": "spmrl_charlm",
    "en": "ptb3-revised_charlm",
    "es": "combined_charlm",
    "id": "icon_charlm",
    "it": "vit_charlm",
    "ja": "alt_charlm",
    "pt": "cintil_charlm",
    #"tr": "starlang_charlm",
    "vi": "vlsp22_charlm",
    "zh-hans": "ctb-51_charlm",
}

optional_constituency = {
    "tr": "starlang_charlm",
}

# an alternate tokenizer for languages which aren't trained from a base UD source
default_tokenizer = {
    "my": "alt",
}

# ideally we would have a less expensive model as the base model
#default_coref = {
#    "en": "ontonotes_roberta-large_finetuned",
#}

optional_coref = {
    "ca": "udcoref_xlm-roberta-lora",
    "cs": "udcoref_xlm-roberta-lora",
    "de": "udcoref_xlm-roberta-lora",
    "en": "udcoref_xlm-roberta-lora",
    "es": "udcoref_xlm-roberta-lora",
    "fr": "udcoref_xlm-roberta-lora",
    "hi": "deeph_muril-large-cased-lora",
    # UD Coref has both nb and nn datasets for Norwegian
    "nb": "udcoref_xlm-roberta-lora",
    "nn": "udcoref_xlm-roberta-lora",
    "pl": "udcoref_xlm-roberta-lora",
    "ru": "udcoref_xlm-roberta-lora",
}

"""
default transformers to use for various languages

we try to document why we choose a particular model in each case
"""
TRANSFORMERS = {
    # We tested three candidate AR models on POS, Depparse, and NER
    #
    # POS: padt dev set scores, AllTags
    # depparse: padt dev set scores, LAS
    # NER: dev scores on a random split of AQMAR, entity scores
    #
    #                                             pos   depparse  ner
    # none (pt & charlm only)                    94.08    83.49  84.19
    # asafaya/bert-base-arabic                   95.10    84.96  85.98
    # aubmindlab/bert-base-arabertv2             95.33    85.28  84.93
    # aubmindlab/araelectra-base-discriminator   95.66    85.83  86.10
    "ar": "aubmindlab/araelectra-base-discriminator",

    # https://huggingface.co/Maltehb/danish-bert-botxo
    # contrary to normal expectations, this hurts F1
    # on a dev split by about 1 F1
    # "da": "Maltehb/danish-bert-botxo",
    #
    # the multilingual bert is a marginal improvement for conparse
    #
    # December 2022 update:
    # there are quite a few Danish transformers available on HuggingFace
    # here are the results of training a constituency parser with adadelta/adamw
    # on each of them:
    #
    # no bert                              0.8245    0.8230
    # alexanderfalk/danbert-small-cased    0.8236    0.8286
    # Geotrend/distilbert-base-da-cased    0.8268    0.8306
    # sarnikowski/convbert-small-da-cased  0.8322    0.8341
    # bert-base-multilingual-cased         0.8341    0.8342
    # vesteinn/ScandiBERT-no-faroese       0.8373    0.8408
    # Maltehb/danish-bert-botxo            0.8383    0.8408
    # vesteinn/ScandiBERT                  0.8421    0.8475
    #
    # Also, two models have token windows too short for use with the
    # Danish dataset:
    #  jonfd/electra-small-nordic
    #  Maltehb/aelaectra-danish-electra-small-cased
    #
    "da": "vesteinn/ScandiBERT",

    # As of April 2022, the bert models available have a weird
    # tokenizer issue where soft hyphen causes it to crash.
    # We attempt to compensate for that in the dev branch
    #
    # NER scores
    #     model                                       dev      text
    # xlm-roberta-large                              86.56    85.23
    # bert-base-german-cased                         87.59    86.95
    # dbmdz/bert-base-german-cased                   88.27    87.47
    # german-nlp-group/electra-base-german-uncased   88.60    87.09
    #
    # constituency scores w/ peft, March 2024 model, in-order
    #    model             dev     test
    #   xlm-roberta-base  95.17   93.34
    #   xlm-roberta-large 95.86   94.46    (!!!)
    #   bert-base         95.24   93.24
    #   dbmdz/bert        95.32   93.33
    #   german/electra    95.72   94.05
    #
    # POS scores
    #    model             dev     test
    #   None              88.65   87.28
    #   xlm-roberta-large 89.21   88.11
    #   bert-base         89.52   88.42
    #   dbmdz/bert        89.67   88.54
    #   german/electra    89.98   88.66
    #
    # depparse scores, LAS
    #    model             dev     test
    #   None              87.76   84.37
    #   xlm-roberta-large 89.00   85.79
    #   bert-base         88.72   85.40
    #   dbmdz/bert        88.70   85.14
    #   german/electra    89.21   86.06
    "de": "german-nlp-group/electra-base-german-uncased",

    # experiments on various forms of roberta & electra
    #  https://huggingface.co/roberta-base
    #  https://huggingface.co/roberta-large
    #  https://huggingface.co/google/electra-small-discriminator
    #  https://huggingface.co/google/electra-base-discriminator
    #  https://huggingface.co/google/electra-large-discriminator
    #
    # experiments using the different models for POS tagging,
    # dev set, including WV and charlm, AllTags score:
    #  roberta-base:   95.67
    #  roberta-large:  95.98
    #  electra-small:  95.31
    #  electra-base:   95.90
    #  electra-large:  96.01
    #
    # depparse scores, dev set, no finetuning, with WV and charlm
    #                   UAS   LAS  CLAS  MLAS  BLEX
    #  roberta-base:   93.16 91.20 89.87 89.38 89.87
    #  roberta-large:  93.47 91.56 90.13 89.71 90.13
    #  electra-small:  92.17 90.02 88.25 87.66 88.25
    #  electra-base:   93.42 91.44 90.10 89.67 90.10
    #  electra-large:  94.07 92.17 90.99 90.53 90.99
    #
    # conparse scores, dev & test set, with WV and charlm
    #  roberta_base:   96.05 95.60
    #  roberta_large:  95.95 95.60
    #  electra-small:  95.33 95.04
    #  electra-base:   96.09 95.98
    #  electra-large:  96.25 96.14
    #
    # conparse scores w/ finetune, dev & test set, with WV and charlm
    #  roberta_base:   96.07 95.81
    #  roberta_large:  96.37 96.41   (!!!)
    #  electra-small:  95.62 95.36
    #  electra-base:   96.21 95.94
    #  electra-large:  96.40 96.32
    #
    "en": "google/electra-large-discriminator",

    # TODO need to test, possibly compare with others
    "es": "bertin-project/bertin-roberta-base-spanish",

    # NER scores for a couple Persian options:
    # none:
    # dev:  2022-04-23 01:44:53 INFO: fa_arman 79.46
    # test: 2022-04-23 01:45:03 INFO: fa_arman 80.06
    #
    # HooshvareLab/bert-fa-zwnj-base
    # dev:  2022-04-23 02:43:44 INFO: fa_arman 80.87
    # test: 2022-04-23 02:44:07 INFO: fa_arman 80.81
    #
    # HooshvareLab/roberta-fa-zwnj-base
    # dev:  2022-04-23 16:23:25 INFO: fa_arman 81.23
    # test: 2022-04-23 16:23:48 INFO: fa_arman 81.11
    #
    # HooshvareLab/bert-base-parsbert-uncased
    # dev:  2022-04-26 10:42:09 INFO: fa_arman 82.49
    # test: 2022-04-26 10:42:31 INFO: fa_arman 83.16
    "fa": 'HooshvareLab/bert-base-parsbert-uncased',

    # NER scores for a couple options:
    # none:
    # dev:  2022-03-04 INFO: fi_turku 83.45
    # test: 2022-03-04 INFO: fi_turku 86.25
    #
    # bert-base-multilingual-cased
    # dev:  2022-03-04 INFO: fi_turku 85.23
    # test: 2022-03-04 INFO: fi_turku 89.00
    #
    # TurkuNLP/bert-base-finnish-cased-v1:
    # dev:  2022-03-04 INFO: fi_turku 88.41
    # test: 2022-03-04 INFO: fi_turku 91.36
    "fi": "TurkuNLP/bert-base-finnish-cased-v1",

    # POS dev set tagging results for French:
    #  No bert:
    #    98.60  100.00   98.55   98.04
    #  dbmdz/electra-base-french-europeana-cased-discriminator
    #    98.70  100.00   98.69   98.24
    #  benjamin/roberta-base-wechsel-french
    #    98.71  100.00   98.75   98.26
    #  camembert/camembert-large
    #    98.75  100.00   98.75   98.30
    #  camembert-base
    #    98.78  100.00   98.77   98.33
    #
    # GSD depparse dev set results for French:
    #  No bert:
    #    95.83 94.52 91.34 91.10 91.34
    #  camembert/camembert-large
    #    96.80 95.71 93.37 93.13 93.37
    #  TODO: the rest of the chart
    "fr": "camembert/camembert-large",

    # Ancient Greek has a surprising number of transformers, considering
    #    Model           POS        Depparse LAS
    # None              0.8812       0.7684
    # Microbert M       0.8883       0.7706
    # Microbert MX      0.8910       0.7755
    # Microbert MXP     0.8916       0.7742
    # Pranaydeeps Bert  0.9139       0.7987
    "grc": "pranaydeeps/Ancient-Greek-BERT",

    # a couple possibilities to experiment with for Hebrew
    # dev scores for POS and depparse
    # https://huggingface.co/imvladikon/alephbertgimmel-base-512
    #   UPOS    XPOS  UFeats AllTags
    #  97.25   97.25   92.84   91.81
    #   UAS   LAS  CLAS  MLAS  BLEX
    #  94.42 92.47 89.49 88.82 89.49
    #
    # https://huggingface.co/onlplab/alephbert-base
    #   UPOS    XPOS  UFeats AllTags
    #  97.37   97.37   92.50   91.55
    #   UAS   LAS  CLAS  MLAS  BLEX
    #  94.06 92.12 88.80 88.13 88.80
    #
    # https://huggingface.co/avichr/heBERT
    #   UPOS    XPOS  UFeats AllTags
    #  97.09   97.09   92.36   91.28
    #   UAS   LAS  CLAS  MLAS  BLEX
    #  94.29 92.30 88.99 88.38 88.99
    "he": "imvladikon/alephbertgimmel-base-512",

    # can also experiment with xlm-roberta
    # on a coref dataset from IITH, span F1:
    #                         dev      test
    #  xlm-roberta-large   0.63635   0.66579
    #  muril-large         0.65369   0.68290
    "hi": "google/muril-large-cased",

    # https://huggingface.co/xlm-roberta-base
    # Scores by entity for armtdp NER on 18 labels:
    # no bert : 86.68
    # xlm-roberta-base : 89.31
    "hy": "xlm-roberta-base",

    # Indonesian POS experiments: dev set of GSD
    # python3 stanza/utils/training/run_pos.py id_gsd --no_bert
    # python3 stanza/utils/training/run_pos.py id_gsd --bert_model ...
    # also ran on the ICON constituency dataset
    #  model                                      POS       CON
    # no_bert                                    89.95     84.74
    # flax-community/indonesian-roberta-large    89.78 (!)  xxx
    # flax-community/indonesian-roberta-base     90.14      xxx
    # indobenchmark/indobert-base-p2             90.09
    # indobenchmark/indobert-base-p1             90.14
    # indobenchmark/indobert-large-p1            90.19
    # indolem/indobert-base-uncased              90.21     88.60
    # cahya/bert-base-indonesian-1.5G            90.32     88.15
    # cahya/roberta-base-indonesian-1.5G         90.40     87.27
    "id": "indolem/indobert-base-uncased",

    # from https://github.com/idb-ita/GilBERTo
    # annoyingly, it doesn't handle cased text
    # supposedly there is an argument "do_lower_case"
    # but that still leaves a lot of unk tokens
    # "it": "idb-ita/gilberto-uncased-from-camembert",
    #
    # from https://github.com/musixmatchresearch/umberto
    # on NER, this gets 88.37 dev and 91.02 test
    # another option is dbmdz/bert-base-italian-cased,
    # which gets 87.27 dev and 90.32 test
    #
    #  in-order constituency parser on the VIT dev set:
    # dbmdz/bert-base-italian-cased                       0.8079
    # dbmdz/bert-base-italian-xxl-cased:                  0.8195
    # Musixmatch/umberto-commoncrawl-cased-v1:            0.8256
    # dbmdz/electra-base-italian-xxl-cased-discriminator: 0.8314
    #
    #  FBK NER dev set:
    # dbmdz/bert-base-italian-cased:                      87.76
    # Musixmatch/umberto-commoncrawl-cased-v1:            88.62
    # dbmdz/bert-base-italian-xxl-cased:                  88.84
    # dbmdz/electra-base-italian-xxl-cased-discriminator: 89.91
    #
    #  combined UD POS dev set:                             UPOS    XPOS  UFeats AllTags
    # dbmdz/bert-base-italian-cased:                       98.62   98.53   98.06   97.49
    # dbmdz/bert-base-italian-xxl-cased:                   98.61   98.54   98.07   97.58
    # dbmdz/electra-base-italian-xxl-cased-discriminator:  98.64   98.54   98.14   97.61
    # Musixmatch/umberto-commoncrawl-cased-v1:             98.56   98.45   98.13   97.62
    "it": "dbmdz/electra-base-italian-xxl-cased-discriminator",

    # for Japanese
    # there are others that would also work,
    # but they require different tokenizers instead of being
    # plug & play
    #
    # Constitutency scores on ALT (in-order)
    # no bert: 90.68 dev, 91.40 test
    # rinna:   91.54 dev, 91.89 test
    "ja": "rinna/japanese-roberta-base",

    # could also try:
    # l3cube-pune/marathi-bert-v2
    #  or
    # https://huggingface.co/l3cube-pune/hindi-marathi-dev-roberta
    # l3cube-pune/hindi-marathi-dev-roberta
    #
    # depparse ufal dev scores:
    #  no transformer              74.89 63.70 57.43 53.01 57.43
    #  l3cube-pune/marathi-roberta 76.48 66.21 61.20 57.60 61.20
    "mr": "l3cube-pune/marathi-roberta",

    # https://huggingface.co/allegro/herbert-base-cased
    # Scores by entity on the NKJP NER task:
    # no bert (dev/test): 88.64/88.75
    # herbert-base-cased (dev/test): 91.48/91.02,
    # herbert-large-cased (dev/test): 92.25/91.62
    # sdadas/polish-roberta-large-v2 (dev/test): 92.66/91.22
    "pl": "allegro/herbert-base-cased",

    # experiments on the cintil conparse dataset
    # ran a variety of transformer settings
    # found the following dev set scores after 400 iterations:
    # Geotrend/distilbert-base-pt-cased : not plug & play
    # no bert: 0.9082
    # xlm-roberta-base: 0.9109
    # xlm-roberta-large: 0.9254
    # adalbertojunior/distilbert-portuguese-cased: 0.9300
    # neuralmind/bert-base-portuguese-cased: 0.9307
    # neuralmind/bert-large-portuguese-cased: 0.9343
    "pt": "neuralmind/bert-large-portuguese-cased",

    # hope is actually to build our own using a large text collection
    "sd": "google/muril-large-cased",

    # Tamil options: quite a few, need to run a bunch of experiments
    #                               dev pos    dev depparse las
    # no transformer                 82.82        69.12
    # ai4bharat/indic-bert           82.98        70.47
    # lgessler/microbert-tamil-mxp   83.21        69.28
    # monsoon-nlp/tamillion          83.37        69.28
    # l3cube-pune/tamil-bert         85.27        72.53
    # d42kw01f/Tamil-RoBERTa         85.59        70.55
    # google/muril-base-cased        85.67        72.68
    # google/muril-large-cased       86.30        72.45
    "ta": "google/muril-large-cased",


    # https://huggingface.co/dbmdz/bert-base-turkish-128k-cased
    # helps the Turkish model quite a bit
    "tr": "dbmdz/bert-base-turkish-128k-cased",

    # from https://github.com/VinAIResearch/PhoBERT
    # "vi": "vinai/phobert-base",
    # using 6 or 7 layers of phobert-large is slightly
    # more effective for constituency parsing than
    # using 4 layers of phobert-base
    # ... going beyond 4 layers of phobert-base
    # does not help the scores
    "vi": "vinai/phobert-large",

    # https://github.com/ymcui/Chinese-BERT-wwm
    # there's also hfl/chinese-roberta-wwm-ext-large
    # or hfl/chinese-electra-base-discriminator
    # or hfl/chinese-electra-180g-large-discriminator,
    #   which works better than the below roberta on constituency
    # "zh-hans": "hfl/chinese-roberta-wwm-ext",
    "zh-hans": "hfl/chinese-electra-180g-large-discriminator",
}

TRANSFORMER_LAYERS = {
    # not clear what the best number is without more experiments,
    # but more than 4 is working better than just 4
    "vi": 7,
}

TRANSFORMER_NICKNAMES = {
    # ar
    "asafaya/bert-base-arabic": "asafaya-bert",
    "aubmindlab/araelectra-base-discriminator": "aubmind-electra",
    "aubmindlab/bert-base-arabertv2": "aubmind-bert",

    # da
    "vesteinn/ScandiBERT": "scandibert",

    # de
    "bert-base-german-cased": "bert-base-german-cased",
    "dbmdz/bert-base-german-cased": "dbmdz-bert-german-cased",
    "german-nlp-group/electra-base-german-uncased": "german-nlp-electra",

    # en
    "bert-base-multilingual-cased": "mbert",
    "xlm-roberta-large": "xlm-roberta-large",
    "google/electra-large-discriminator": "electra-large",
    "microsoft/deberta-v3-large": "deberta-v3-large",

    # es
    "bertin-project/bertin-roberta-base-spanish": "bertin-roberta",

    # fa
    "HooshvareLab/bert-base-parsbert-uncased": "parsbert",

    # fi
    "TurkuNLP/bert-base-finnish-cased-v1": "bert",

    # fr
    "benjamin/roberta-base-wechsel-french": "wechsel-roberta",
    "camembert-base": "camembert-base",
    "camembert/camembert-large": "camembert-large",
    "dbmdz/electra-base-french-europeana-cased-discriminator": "dbmdz-electra",

    # grc
    "pranaydeeps/Ancient-Greek-BERT": "grc-pranaydeeps",
    "lgessler/microbert-ancient-greek-m": "grc-microbert-m",
    "lgessler/microbert-ancient-greek-mx": "grc-microbert-mx",
    "lgessler/microbert-ancient-greek-mxp": "grc-microbert-mxp",
    "altsoph/bert-base-ancientgreek-uncased": "grc-altsoph",

    # he
    "imvladikon/alephbertgimmel-base-512" : "alephbertgimmel",

    # hy
    "xlm-roberta-base": "xlm-roberta-base",

    # id
    "indolem/indobert-base-uncased":         "indobert",
    "indobenchmark/indobert-large-p1":       "indobenchmark-large-p1",
    "indobenchmark/indobert-base-p1":        "indobenchmark-base-p1",
    "indobenchmark/indobert-lite-large-p1":  "indobenchmark-lite-large-p1",
    "indobenchmark/indobert-lite-base-p1":   "indobenchmark-lite-base-p1",
    "indobenchmark/indobert-large-p2":       "indobenchmark-large-p2",
    "indobenchmark/indobert-base-p2":        "indobenchmark-base-p2",
    "indobenchmark/indobert-lite-large-p2":  "indobenchmark-lite-large-p2",
    "indobenchmark/indobert-lite-base-p2":   "indobenchmark-lite-base-p2",

    # it
    "dbmdz/electra-base-italian-xxl-cased-discriminator": "electra",

    # ja
    "rinna/japanese-roberta-base": "rinna-roberta",

    # mr
    "l3cube-pune/marathi-roberta": "l3cube-marathi-roberta",

    # pl
    "allegro/herbert-base-cased": "herbert",

    # pt
    "neuralmind/bert-large-portuguese-cased": "bertimbau",

    # ta: tamil
    "monsoon-nlp/tamillion":         "tamillion",
    "lgessler/microbert-tamil-m":    "ta-microbert-m",
    "lgessler/microbert-tamil-mxp":  "ta-microbert-mxp",
    "l3cube-pune/tamil-bert":        "l3cube-tamil-bert",
    "d42kw01f/Tamil-RoBERTa":        "ta-d42kw01f-roberta",

    # tr
    "dbmdz/bert-base-turkish-128k-cased": "bert",

    # vi
    "vinai/phobert-base": "phobert-base",
    "vinai/phobert-large": "phobert-large",

    # zh
    "google-bert/bert-base-chinese": "google-bert-chinese",
    "hfl/chinese-roberta-wwm-ext": "hfl-roberta-chinese",
    "hfl/chinese-electra-180g-large-discriminator": "electra-large",

    # multi-lingual Indic
    "ai4bharat/indic-bert": "indic-bert",
    "google/muril-base-cased": "muril-base-cased",
    "google/muril-large-cased": "muril-large-cased",
}

def known_nicknames():
    """
    Return a list of all the transformer nicknames

    We return a list so that we can sort them in decreasing key length
    """
    nicknames = list(value for key, value in TRANSFORMER_NICKNAMES.items())

    # previously unspecific transformers get "transformer" as the nickname
    nicknames.append("transformer")

    nicknames = sorted(nicknames, key=lambda x: -len(x))

    return nicknames
