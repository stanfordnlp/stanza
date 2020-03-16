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
  "da": "ddt",
  "nl": "alpino",
  "en": "ewt",
  "et": "edt",
  "fi": "tdt",
  "fr": "gsd",
  "gl": "ctg",
  "de": "gsd",
  "got": "proiel",
  "el": "gdt",
  "he": "htb",
  "hi": "hdtb",
  "hu": "szeged",
  "id": "gsd",
  "ga": "idt",
  "it": "isdt",
  "ja": "gsd",
  "kk": "ktb",
  "ko": "kaist",
  "kmr": "mg",
  "la": "ittb",
  "lv": "lvtb",
  "sme": "giella",
  "cu": "proiel",
  "fro": "srcmf",
  "fa": "seraji",
  "pl": "lfg",
  "pt": "bosque",
  "ro": "rrt",
  "ru": "syntagrus",
  "sr": "set",
  "sk": "snk",
  "sl": "ssj",
  "es": "ancora",
  "sv": "talbanken",
  "tr": "imst",
  "uk": "iu",
  "hsb": "ufal",
  "ur": "udtb",
  "ug": "udt",
  "vi": "vtb",
  "lt": "hse",
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
  "ar": "aqmar",
  "de": "conll03",
  "en": "ontonotes",
  "es": "conll02",
  "fr": "wikiner",
  "nl": "conll02",
  "ru": "wikiner",
  "zh-hans": "ontonotes"
}


# default charlms for languages
default_charlms = {
  "ar": "ccwiki",
  "de": "newswiki",
  "en": "1billion",
  "es": "newswiki",
  "fr": "newswiki",
  "nl": "ccwiki",
  "ru": "newswiki",
  "zh-hans": "gigaword"
}


# map processor name to file ending
processor_to_ending = {
  "tokenize": "tokenizer",
  "mwt": "mwt_expander",
  "pos": "tagger",
  "lemma": "lemmatizer",
  "depparse": "parser",
  "ner": "nertagger",
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
    "orv": "Old_Russian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
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
    "uk": "Ukrainian",
    "hsb": "Upper_Sorbian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "vi": "Vietnamese",
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


def process_dirs(args):
    dirs = sorted(os.listdir(args.input_dir))
    resources = {}

    for dir in dirs:
        print(dir)
        models = sorted(os.listdir(os.path.join(args.input_dir, dir)))
        for model in models:
            if not model.endswith('.pt'): continue
            # get processor
            lang, package, processor = model.replace('.pt', '').replace('.', '_').split('_', 2)
            processor = ending_to_processor[processor]
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
                charlm_package = default_charlms[lang]
                dependencies = [{'model': 'forward_charlm', 'package': charlm_package}, {'model': 'backward_charlm', 'package': charlm_package}]
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
            print(lang + ' not in default treebanks!!!')
            continue
        print(lang)

        ud_package = default_treebanks[lang]
        os.chdir(os.path.join(args.output_dir, lang))
        zipf = zipfile.ZipFile('default.zip', 'w', zipfile.ZIP_DEFLATED)
        default_processors = {}
        default_dependencies = {'pos': [{'model': 'pretrain', 'package': ud_package}], 'depparse': [{'model': 'pretrain', 'package': ud_package}]}

        if lang in default_ners:
            ner_package = default_ners[lang]
            charlm_package = default_charlms[lang]
            default_dependencies['ner'] = [{'model': 'forward_charlm', 'package': charlm_package}, {'model': 'backward_charlm', 'package': charlm_package}]
        
        processors = ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner', 'pretrain', 'forward_charlm', 'backward_charlm'] if lang in default_ners else ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'pretrain']
        for processor in processors:
            if processor == 'ner': package = ner_package
            elif processor in ['forward_charlm', 'backward_charlm']: package = charlm_package
            else: package = ud_package

            if os.path.exists(os.path.join(args.output_dir, lang, processor, package + '.pt')):
                if processor in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner']:
                     default_processors[processor] = package
                zipf.write(processor)
                zipf.write(os.path.join(processor, package + '.pt'))
        zipf.close()
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
