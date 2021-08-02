import json
import argparse
import os
from pathlib import Path
import hashlib
import shutil
import zipfile

# default treebank for languages
default_treebanks = {
  "af": "afribooms",
  "grc": "proiel",
  "ar": "padt",
  "hy": "armtdp",
  "eu": "bdt",
  "bg": "btb",
  "bxr": "bdt",
  "ca": "ancora",
  "zh-hant": "gsd",
  "hr": "set",
  "cs": "pdt",
  "cy": "ccg",
  "da": "ddt",
  "nl": "alpino",
  "en": "combined",
  "et": "edt",
  "fi": "tdt",
  "fo": "farpahc",
  "fr": "gsd",
  "gl": "ctg",
  "de": "gsd",
  "got": "proiel",
  "el": "gdt",
  "he": "htb",
  "hi": "hdtb",
  "hu": "szeged",
  "id": "gsd",
  "is": "icepahc",
  "ga": "idt",
  "it": "combined",
  "ja": "gsd",
  "kk": "ktb",
  "ko": "kaist",
  "kmr": "mg",
  "la": "ittb",
  "lv": "lvtb",
  "pcm": "nsc",
  "sme": "giella",
  "cu": "proiel",
  "fro": "srcmf",
  "fa": "perdt",
  "pl": "pdb",
  "pt": "bosque",
  "ro": "rrt",
  "ru": "syntagrus",
  "sa": "vedic",
  "sr": "set",
  "sk": "snk",
  "sl": "ssj",
  "es": "ancora",
  "sv": "talbanken",
  "th": "orchid",
  "tr": "imst",
  "qtd": "sagt",
  "uk": "iu",
  "hsb": "ufal",
  "ur": "udtb",
  "ug": "udt",
  "vi": "vtb",
  "lt": "alksnis",
  "hyw": "armtdp",
  "wo": "wtb",
  "nb": "bokmaal",
  "mt": "mudt",
  "swl": "sslc",
  "cop": "scriptorium",
  "be": "hse",
  "zh-hans": "gsdsimp",
  "lzh": "kyoto",
  "gd": "arcosg",
  "olo": "kkpp",
  "ta": "ttb",
  "te": "mtg",
  "orv": "torot",
  "nn": "nynorsk",
  "mr": "ufal"
}


# default ner for languages
default_ners = {
  "af": "nchlt",
  "ar": "aqmar",
  "bg": "bsnlp19",
  "de": "conll03",
  "en": "ontonotes",
  "es": "conll02",
  "fi": "turku",
  "fr": "wikiner",
  "hu": "combined",
  "it": "fbk",
  "nl": "conll02",
  "ru": "wikiner",
  "uk": "languk",
  "vi": "vlsp",
  "zh-hans": "ontonotes",
}


# default charlms for languages
default_charlms = {
  "af": "oscar",
  "ar": "ccwiki",
  "bg": "conll17",
  "de": "newswiki",
  "en": "1billion",
  "es": "newswiki",
  "fi": "conll17",
  "fr": "newswiki",
  "it": "conll17",
  "nl": "ccwiki",
  "ru": "newswiki",
  "vi": "conll17",
  "zh-hans": "gigaword"
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
  "uk": {
    "languk": None,
  },
  "hu": {
    "combined": None,
  },
}

# a few languages have sentiment classifier models
default_sentiment = {
  "en": "sstplus",
  "de": "sb10k",
  "vi": "vsfc",
  "zh-hans": "ren",
}

allowed_empty_languages = [
  # we don't have a lot of Thai support yet
  "th"
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
  "pretrain": "pretrain",
  "forward_charlm": "forward_charlm",
  "backward_charlm": "backward_charlm"
}
ending_to_processor = {j: i for i, j in processor_to_ending.items()}

# add full language name to language code and add alias in resources
lcode2lang = {
    "af": "Afrikaans",
    "grc": "Ancient_Greek",
    "ar": "Arabic",
    "hy": "Armenian",
    "eu": "Basque",
    "be": "Belarusian",
    "br": "Breton",
    "bg": "Bulgarian",
    "bxr": "Buryat",
    "ca": "Catalan",
    "zh-hant": "Traditional_Chinese",
    "lzh": "Classical_Chinese",
    "cop": "Coptic",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
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
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "ko": "Korean",
    "kmr": "Kurmanji",
    "lt": "Lithuanian",
    "olo": "Livvi",
    "la": "Latin",
    "lv": "Latvian",
    "mt": "Maltese",
    "mr": "Marathi",
    "pcm": "Naija",
    "sme": "North_Sami",
    "nb": "Norwegian_Bokmaal",
    "nn": "Norwegian_Nynorsk",
    "cu": "Old_Church_Slavonic",
    "fro": "Old_French",
    "orv": "Old_East_Slavic",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "gd": "Scottish_Gaelic",
    "sr": "Serbian",
    "zh-hans": "Simplified_Chinese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "swl": "Swedish_Sign_Language",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "qtd": "Turkish_German",
    "uk": "Ukrainian",
    "hsb": "Upper_Sorbian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "hyw": "Western_Armenian",
    "wo": "Wolof"
}


def ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    ensure_dir(Path(dst).parent)
    shutil.copy(src, dst)


def get_md5(path):
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input dir for various models.')
    parser.add_argument('--output_dir', type=str, help='Output dir for various models.')
    args = parser.parse_args()
    return args


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

def get_ner_dependencies(lang, package):
    if lang not in ner_charlms or package not in ner_charlms[lang]:
        charlm_package = default_charlms[lang]
    else:
        charlm_package = ner_charlms[lang][package]

    if charlm_package is None:
        return None
    else:
        return [{'model': 'forward_charlm', 'package': charlm_package},
                {'model': 'backward_charlm', 'package': charlm_package}]

def process_dirs(args):
    dirs = sorted(os.listdir(args.input_dir))
    resources = {}

    for dir in dirs:
        print(f"Processing models in {dir}")
        models = sorted(os.listdir(os.path.join(args.input_dir, dir)))
        for model in models:
            if not model.endswith('.pt'): continue
            # get processor
            lang, package, processor = split_model_name(model)
            # copy file
            input_path = os.path.join(args.input_dir, dir, model)
            output_path = os.path.join(args.output_dir, lang, processor, package + '.pt')
            ensure_dir(Path(output_path).parent)
            shutil.copy(input_path, output_path)
            # maintain md5
            md5 = get_md5(output_path)
            # maintain dependencies
            if processor == 'pos' or processor == 'depparse':
                dependencies = [{'model': 'pretrain', 'package': package}]
            elif processor == 'ner':
                dependencies = get_ner_dependencies(lang, package)
            elif processor == 'sentiment':
                # so far, this invariant is true:
                # sentiment models use the default pretrain for the language
                pretrain_package = default_treebanks[lang]
                dependencies = [{'model': 'pretrain', 'package': pretrain_package}]
            else:
                dependencies = None
            # maintain resources
            if lang not in resources: resources[lang] = {}
            if processor not in resources[lang]: resources[lang][processor] = {}
            if dependencies:
                resources[lang][processor][package] = {'md5': md5, 'dependencies': dependencies}
            else:
                resources[lang][processor][package] = {'md5': md5}
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
        default_dependencies = {'pos': [{'model': 'pretrain', 'package': ud_package}],
                                'depparse': [{'model': 'pretrain', 'package': ud_package}]}

        if lang in default_ners:
            ner_package = default_ners[lang]
        if lang in default_charlms:
            charlm_package = default_charlms[lang]
        if lang in default_sentiment:
            sentiment_package = default_sentiment[lang]

        if lang in default_ners and lang in default_charlms:
            ner_dependencies = get_ner_dependencies(lang, ner_package)
            if ner_dependencies is not None:
                default_dependencies['ner'] = ner_dependencies
        if lang in default_sentiment:
            # All of the sentiment models created so far have used the default pretrain
            default_dependencies['sentiment'] = [{'model': 'pretrain', 'package': ud_package}]

        processors = ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'pretrain']
        if lang in default_ners:
            processors.append('ner')
        if lang in default_charlms:
            processors.extend(['forward_charlm', 'backward_charlm'])
        if lang in default_sentiment:
            processors.append('sentiment')

        with zipfile.ZipFile('default.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for processor in processors:
                if processor == 'ner': package = ner_package
                elif processor in ['forward_charlm', 'backward_charlm']: package = charlm_package
                elif processor == 'sentiment': package = sentiment_package
                else: package = ud_package

                filename = os.path.join(args.output_dir, lang, processor, package + '.pt')
                if os.path.exists(filename):
                    print("   Model {} package {}: file {}".format(processor, package, filename))
                    if processor in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner', 'sentiment']:
                        default_processors[processor] = package
                    zipf.write(processor)
                    zipf.write(os.path.join(processor, package + '.pt'))
                elif lang in allowed_empty_languages:
                    # we don't have a lot of Thai support yet
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

    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def process_lcode(args):
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    resources_new = {}
    for lang in resources:
        if lang not in lcode2lang:
            print(lang + ' not found in lcode2lang!')
            continue
        lang_name = lcode2lang[lang]
        resources[lang]['lang_name'] = lang_name
        resources_new[lang.lower()] = resources[lang.lower()]
        resources_new[lang_name.lower()] = {'alias': lang.lower()}
    json.dump(resources_new, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def process_misc(args):
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    resources['no'] = {'alias': 'nb'}
    resources['zh'] = {'alias': 'zh-hans'}
    resources['url'] = 'http://nlp.stanford.edu/software/stanza'
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)


def main():
    args = parse_args()
    process_dirs(args)
    process_defaults(args)
    process_lcode(args)
    process_misc(args)


if __name__ == '__main__':
    main()
