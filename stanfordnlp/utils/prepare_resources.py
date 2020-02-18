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
  "zh": "gsd",
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
  "no": "bokmaal",
  "mt": "mudt",
  "swl": "sslc",
  "cop": "scriptorium",
  "be": "hse",
  "zhs": "gsdsimp",
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
  "ar": "ontonotes",
  "de": "conll03",
  "en": "ontonotes",
  "es": "conll02",
  "fr": "wikiner",
  "nl": "conll02",
  "ru": "wikiner",
  "zh": "ontonotes"
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
  "zh": "gigaword"
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
    parser.add_argument('--input_dir', type=str, default='/u/scr/zyh/develop/stanfordnlp-train/saved_models', help='Input dir for various models.')
    parser.add_argument('--output_dir', type=str, default='/u/apache/htdocs/static/software/stanza/output', help='Output dir for various models.')
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


def main():
    args = parse_args()
    process_dirs(args)
    process_defaults(args)


if __name__ == '__main__':
    main()