"""
Converts a directory of models organized by type into a directory organized by language.

Also produces the resources.json file.

For example, on the cluster, you can do this:

python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models --output_dir /u/nlp/software/stanza/models/1.4.1 > resources.out 2>&1
"""

import json
import argparse
import os
from pathlib import Path
import hashlib
import shutil
import zipfile

from stanza.models.common.constant import lcode2lang, two_to_three_letters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/u/nlp/software/stanza/current-models", help='Input dir for various models.  Defaults to the recommended home on the nlp cluster')
    parser.add_argument('--output_dir', type=str, default="/u/nlp/software/stanza/built-models", help='Output dir for various models.')
    args = parser.parse_args()
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    return args


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
    "pl":      "pdb",
    "pt":      "bosque",
    "ro":      "rrt",
    "ru":      "syntagrus",
    "sa":      "vedic",
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

default_pretrains = dict(default_treebanks)
for lang in no_pretrain_languages:
    default_pretrains.pop(lang, None)

# default ner for languages
default_ners = {
    "af": "nchlt",
    "ar": "aqmar",
    "bg": "bsnlp19",
    "da": "ddt",
    "de": "germeval2014",
    "en": "ontonotes",
    "es": "conll02",
    "fa": "arman",
    "fi": "turku",
    "fr": "wikiner",
    "hu": "combined",
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
    "sv": "suc3shuffle",
    "th": "lst20",
    "tr": "starlang",
    "uk": "languk",
    "vi": "vlsp",
    "zh-hans": "ontonotes",
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
    "it": "conll17",
    "ja": "conll17",
    "kk": "oscar",
    "mr": "l3cube",
    "my": "oscar",
    "nb": "conll17",
    "nl": "ccwiki",
    "pl": "oscar",
    "ru": "newswiki",
    "sv": "conll17",
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
        "lst20": "fasttext",
    },
    "zh-hans": {
        "ontonotes": "fasttextwiki",
    }
}

# a few languages have sentiment classifier models
default_sentiment = {
    "en": "sstplus",
    "de": "sb10k",
    "es": "tass2020",
    "mr": "l3cube",
    "vi": "vsfc",
    "zh-hans": "ren",
}

# also, a few languages (very few, currently) have constituency parser models
default_constituency = {
    "da": "arboretum",
    "en": "wsj",
    "es": "combined",
    "it": "turin",
    "ja": "alt",
    "pt": "cintil",
    "tr": "starlang",
    "zh-hans": "ctb",
}

# an alternate tokenizer for languages which aren't trained from a base UD source
default_tokenizer = {
    "my": "alt",
}

allowed_empty_languages = [
    # we don't have a lot of Thai support yet
    "th",
    # only tokenize and NER for Myanmar right now (soon...)
    "my",
]

# map processor name to file ending
processor_to_ending = {
    "tokenize": "tokenizer",
    "mwt": "mwt_expander",
    "pos": "tagger",
    "lemma": "lemmatizer",
    "depparse": "parser",
    "ner": "nertagger",
    "sentiment": "sentiment",
    "constituency": "constituency",
    "pretrain": "pretrain",
    "forward_charlm": "forward_charlm",
    "backward_charlm": "backward_charlm",
    "langid": "langid"
}
ending_to_processor = {j: i for i, j in processor_to_ending.items()}

def ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)


def get_md5(path):
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()


def split_model_name(model):
    """
    Split model names by _

    Takes into account packages with _ and processor types with _
    """
    model = model[:-3].replace('.', '_')
    # sort by key length so that nertagger is checked before tagger, for example
    for processor in sorted(ending_to_processor.keys(), key=lambda x: -len(x)):
        if model.endswith(processor):
            model = model[:-(len(processor)+1)]
            processor = ending_to_processor[processor]
            break
    else:
        raise AssertionError(f"Could not find a processor type in {model}")
    lang, package = model.split('_', 1)
    return lang, package, processor

def get_con_dependencies(lang, package):
    # so far, this invariant is true:
    # constituency models use the default pretrain and charlm for the language
    pretrain_package = default_treebanks[lang]
    dependencies = [{'model': 'pretrain', 'package': pretrain_package}]

    # sometimes there is no charlm for a language that has constituency, though
    charlm_package = default_charlms.get(lang, None)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies

def get_pos_dependencies(lang, package):
    # TODO: group pretrains by the type of pretrain
    # that will greatly cut down on the number of number of copies of
    # pretrains we have floating around
    if lang in no_pretrain_languages:
        dependencies = []
    else:
        dependencies = [{'model': 'pretrain', 'package': package}]

    if lang in pos_charlms and package in pos_charlms[lang]:
        charlm_package = pos_charlms[lang][package]
    else:
        charlm_package = default_charlms.get(lang, None)

    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies

def get_ner_dependencies(lang, package):
    dependencies = []

    if lang not in ner_pretrains or package not in ner_pretrains[lang]:
        pretrain_package = default_treebanks[lang]
    else:
        pretrain_package = ner_pretrains[lang][package]
    if pretrain_package is not None:
        dependencies = [{'model': 'pretrain', 'package': pretrain_package}]

    if lang not in ner_charlms or package not in ner_charlms[lang]:
        charlm_package = default_charlms[lang]
    else:
        charlm_package = ner_charlms[lang][package]

    if charlm_package is not None:
        dependencies = dependencies + [{'model': 'forward_charlm', 'package': charlm_package},
                                       {'model': 'backward_charlm', 'package': charlm_package}]
    return dependencies

def get_sentiment_dependencies(lang, package):
    """
    Return a list of dependencies for the sentiment model

    Generally this will be pretrain, forward & backward charlm
    So far, this invariant is true:
    sentiment models use the default pretrain for the language
    also, they all use the default charlm for a language
    """
    pretrain_package = default_treebanks[lang]
    dependencies = [{'model': 'pretrain', 'package': pretrain_package}]

    charlm_package = default_charlms.get(lang, None)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies


def process_dirs(args):
    dirs = sorted(os.listdir(args.input_dir))
    resources = {}

    for model_dir in dirs:
        print(f"Processing models in {model_dir}")
        models = sorted(os.listdir(os.path.join(args.input_dir, model_dir)))
        for model in models:
            if not model.endswith('.pt'): continue
            # get processor
            lang, package, processor = split_model_name(model)
            # copy file
            input_path = os.path.join(args.input_dir, model_dir, model)
            output_path = os.path.join(args.output_dir, lang, processor, package + '.pt')
            copy_file(input_path, output_path)
            # maintain md5
            md5 = get_md5(output_path)
            # maintain dependencies
            dependencies = None
            if processor == 'depparse':
                if lang not in no_pretrain_languages:
                    dependencies = [{'model': 'pretrain', 'package': package}]
            elif processor == 'pos':
                dependencies = get_pos_dependencies(lang, package)
            elif processor == 'ner':
                dependencies = get_ner_dependencies(lang, package)
            elif processor == 'sentiment':
                dependencies = get_sentiment_dependencies(lang, package)
            elif processor == 'constituency':
                dependencies = get_con_dependencies(lang, package)
            # maintain resources
            if lang not in resources: resources[lang] = {}
            if processor not in resources[lang]: resources[lang][processor] = {}
            if dependencies:
                resources[lang][processor][package] = {'md5': md5, 'dependencies': dependencies}
            else:
                resources[lang][processor][package] = {'md5': md5}
    print("Processed initial model directories.  Writing preliminary resources.json")
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def process_defaults(args):
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    for lang in resources:
        if lang not in default_treebanks: 
            raise AssertionError(f'{lang} not in default treebanks!!!')
        print(f'Preparing default models for language {lang}')

        ud_package = default_treebanks[lang]
        os.chdir(os.path.join(args.output_dir, lang))
        default_processors = {}
        if lang in allowed_empty_languages or lang in no_pretrain_languages:
            default_dependencies = {}
        else:
            default_dependencies = {'pos': get_pos_dependencies(lang, ud_package),
                                    'depparse': [{'model': 'pretrain', 'package': ud_package}]}

        if lang in default_ners:
            ner_package = default_ners[lang]
        if lang in default_charlms:
            charlm_package = default_charlms[lang]
        if lang in default_ners and lang in default_charlms:
            ner_dependencies = get_ner_dependencies(lang, ner_package)
            if ner_dependencies is not None:
                default_dependencies['ner'] = ner_dependencies
        if lang in default_sentiment:
            sentiment_package = default_sentiment[lang]
            sentiment_dependencies = get_sentiment_dependencies(lang, package)
            default_dependencies['sentiment'] = sentiment_dependencies
        if lang in default_constituency:
            constituency_package = default_constituency[lang]
            default_dependencies['constituency'] = get_con_dependencies(lang, constituency_package)

        processors = ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'pretrain']
        if lang in default_ners:
            processors.append('ner')
        if lang in default_charlms:
            processors.extend(['forward_charlm', 'backward_charlm'])
        if lang in default_sentiment:
            processors.append('sentiment')
        if lang in default_constituency:
            processors.append('constituency')

        if lang == 'multilingual':
            processors = ['langid']
            default_dependencies = {}

        with zipfile.ZipFile('default.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for processor in processors:
                if processor == 'ner': package = ner_package
                elif processor in ['forward_charlm', 'backward_charlm']: package = charlm_package
                elif processor == 'sentiment': package = sentiment_package
                elif processor == 'constituency': package = constituency_package
                elif processor == 'langid': package = 'ud' 
                elif processor == 'tokenize' and lang in default_tokenizer: package = default_tokenizer[lang]
                else: package = ud_package

                filename = os.path.join(args.output_dir, lang, processor, package + '.pt')

                if os.path.exists(filename):
                    print("   Model {} package {}: file {}".format(processor, package, filename))
                    if processor in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner', 'sentiment', 'constituency', 'langid']:
                        default_processors[processor] = package
                    zipf.write(os.path.join(processor, package + '.pt'))
                elif lang in allowed_empty_languages:
                    # we don't have a lot of Thai or Myanmar support yet
                    pass
                elif processor == 'lemma':
                    # a few languages use the identity lemmatizer -
                    # there might be a better way to encode that here
                    default_processors[processor] = "identity"
                    print(" --Model {} package {}: no file {}, assuming identity lemmatizer".format(processor, package, filename))
                elif processor in ('mwt', 'pretrain'):
                    # some languages don't have MWT, so skip ig
                    # others have pos and depparse built with no pretrain
                    print(" --Model {} package {}: no file {}, skipping".format(processor, package, filename))
                else:
                    raise FileNotFoundError(f"Could not find an expected model file for {lang} {processor} {package} : {filename}")
        default_md5 = get_md5(os.path.join(args.output_dir, lang, 'default.zip'))
        resources[lang]['default_processors'] = default_processors
        resources[lang]['default_dependencies'] = default_dependencies
        resources[lang]['default_md5'] = default_md5

    print("Processed default model dependencies.  Writing resources.json")
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def process_lcode(args):
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    resources_new = {}
    resources_new["multilingual"] = resources["multilingual"]
    for lang in resources:
        if lang == 'multilingual':
            continue
        if lang not in lcode2lang:
            print(lang + ' not found in lcode2lang!')
            continue
        lang_name = lcode2lang[lang]
        resources[lang]['lang_name'] = lang_name
        resources_new[lang.lower()] = resources[lang.lower()]
        resources_new[lang_name.lower()] = {'alias': lang.lower()}
        if lang.lower() in two_to_three_letters:
            resources_new[two_to_three_letters[lang.lower()]] = {'alias': lang.lower()}
    print("Processed lcode aliases.  Writing resources.json")
    json.dump(resources_new, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def process_misc(args):
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    resources['no'] = {'alias': 'nb'}
    resources['zh'] = {'alias': 'zh-hans'}
    resources['url'] = 'https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/v{resources_version}/models/{filename}'
    print("Finalized misc attributes.  Writing resources.json")
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def main():
    args = parse_args()
    process_dirs(args)
    process_defaults(args)
    process_lcode(args)
    process_misc(args)


if __name__ == '__main__':
    main()

