"""
Converts a directory of models organized by type into a directory organized by language.

Also produces the resources.json file.

For example, on the cluster, you can do this:

python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models-1.5.0 --output_dir /u/nlp/software/stanza/models/1.5.0 > resources.out 2>&1
nlprun -a stanza-1.2 -q john "python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models-1.5.0 --output_dir /u/nlp/software/stanza/models/1.5.0" -o resources.out
"""

import argparse
import json
import os
from pathlib import Path
import hashlib
import shutil
import zipfile

from stanza import __resources_version__
from stanza.models.common.constant import lcode2lang, two_to_three_letters, three_to_two_letters
from stanza.resources.default_packages import default_treebanks, no_pretrain_languages, default_pretrains, pos_pretrains, depparse_pretrains, ner_pretrains, default_charlms, pos_charlms, depparse_charlms, ner_charlms, lemma_charlms, known_nicknames
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/u/nlp/software/stanza/models/current-models-%s" % __resources_version__, help='Input dir for various models.  Defaults to the recommended home on the nlp cluster')
    parser.add_argument('--output_dir', type=str, default="/u/nlp/software/stanza/models/%s" % __resources_version__, help='Output dir for various models.')
    args = parser.parse_args()
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    return args


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
    "en": "sstplus",
    "de": "sb10k",
    "es": "tass2020",
    "mr": "l3cube",
    "vi": "vsfc",
    "zh-hans": "ren",
}

# also, a few languages (very few, currently) have constituency parser models
default_constituency = {
    "da": "arboretum_charlm",
    "en": "ptb3-revised_charlm",
    "es": "combined_charlm",
    "id": "icon_charlm",
    "it": "vit_charlm",
    "ja": "alt_charlm",
    "pt": "cintil_nocharlm",
    #"tr": "starlang_charlm",
    "vi": "vlsp22_charlm",
    "zh-hans": "ctb-51_charlm",
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
    # currently only tokenize and NER for SD as well
    "sd",
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

def split_package(package):
    if package.endswith("_finetuned"):
        package = package[:-10]

    if package.endswith("_nopretrain"):
        package = package[:-11]
        return package, False, False
    if package.endswith("_nocharlm"):
        package = package[:-9]
        return package, True, False
    if package.endswith("_charlm"):
        package = package[:-7]
        return package, True, True
    for nickname in known_nicknames():
        if package.endswith(nickname):
            # +1 for the underscore
            package = package[:-(len(nickname)+1)]
            return package, True, True

    # guess it was a model which wasn't built with the new naming convention of putting the pretrain type at the end
    # assume WV and charlm... if the language / package doesn't allow for one, that should be caught later
    return package, True, True

def get_pretrain_package(lang, package, model_pretrains, default_pretrains):
    package, uses_pretrain, _ = split_package(package)

    if not uses_pretrain or lang in no_pretrain_languages:
        return None
    elif model_pretrains is not None and lang in model_pretrains and package in model_pretrains[lang]:
        return model_pretrains[lang][package]
    elif lang in default_pretrains:
        return default_pretrains[lang]

    raise RuntimeError("pretrain not specified for lang %s package %s" % (lang, package))

def get_charlm_package(lang, package, model_charlms, default_charlms):
    package, _, uses_charlm = split_package(package)

    if not uses_charlm:
        return None

    if model_charlms is not None and lang in model_charlms and package in model_charlms[lang]:
        return model_charlms[lang][package]
    else:
        return default_charlms.get(lang, None)

def get_con_dependencies(lang, package):
    # so far, this invariant is true:
    # constituency models use the default pretrain and charlm for the language
    # sometimes there is no charlm for a language that has constituency, though
    pretrain_package = get_pretrain_package(lang, package, None, default_pretrains)
    dependencies = [{'model': 'pretrain', 'package': pretrain_package}]

    charlm_package = default_charlms.get(lang, None)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies

def get_pos_charlm_package(lang, package):
    return get_charlm_package(lang, package, pos_charlms, default_charlms)

def get_pos_dependencies(lang, package):
    dependencies = []

    pretrain_package = get_pretrain_package(lang, package, pos_pretrains, default_pretrains)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})

    charlm_package = get_pos_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies

def get_lemma_charlm_package(lang, package):
    return get_charlm_package(lang, package, lemma_charlms, default_charlms)

def get_lemma_dependencies(lang, package):
    dependencies = []

    charlm_package = get_lemma_charlm_package(lang, package)

    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies


def get_depparse_charlm_package(lang, package):
    return get_charlm_package(lang, package, depparse_charlms, default_charlms)

def get_depparse_dependencies(lang, package):
    dependencies = []

    pretrain_package = get_pretrain_package(lang, package, depparse_pretrains, default_pretrains)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})

    charlm_package = get_depparse_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies

def get_ner_charlm_package(lang, package):
    return get_charlm_package(lang, package, ner_charlms, default_charlms)

def get_ner_dependencies(lang, package):
    dependencies = []

    pretrain_package = get_pretrain_package(lang, package, ner_pretrains, default_pretrains)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})

    charlm_package = get_ner_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies

def get_sentiment_dependencies(lang, package):
    """
    Return a list of dependencies for the sentiment model

    Generally this will be pretrain, forward & backward charlm
    So far, this invariant is true:
    sentiment models use the default pretrain for the language
    also, they all use the default charlm for a language
    """
    pretrain_package = get_pretrain_package(lang, package, None, default_pretrains)
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
        for model in tqdm(models):
            if not model.endswith('.pt'): continue
            # get processor
            lang, package, processor = split_model_name(model)
            # copy file
            input_path = os.path.join(args.input_dir, model_dir, model)
            output_path = os.path.join(args.output_dir, lang, "models", processor, package + '.pt')
            copy_file(input_path, output_path)
            # maintain md5
            md5 = get_md5(output_path)
            # maintain dependencies
            dependencies = None
            if processor == 'depparse':
                dependencies = get_depparse_dependencies(lang, package)
            elif processor == 'lemma':
                dependencies = get_lemma_dependencies(lang, package)
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

def get_default_pos_package(lang, ud_package):
    charlm_package = get_pos_charlm_package(lang, ud_package)
    if charlm_package is not None:
        return ud_package + "_charlm"
    if lang in no_pretrain_languages:
        return ud_package + "_nopretrain"
    return ud_package + "_nocharlm"

def get_default_depparse_package(lang, ud_package):
    charlm_package = get_depparse_charlm_package(lang, ud_package)
    if charlm_package is not None:
        return ud_package + "_charlm"
    if lang in no_pretrain_languages:
        return ud_package + "_nopretrain"
    return ud_package + "_nocharlm"

def process_defaults(args):
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    for lang in resources:
        if all(k in ("backward_charlm", "forward_charlm", "pretrain") for k in resources[lang].keys()):
            print(f'Skipping empty resources for language {lang}')
            continue
        if lang not in default_treebanks: 
            raise AssertionError(f'{lang} not in default treebanks!!!')
        print(f'Preparing default models for language {lang}')

        pretrains_needed = set()

        ud_package = default_treebanks[lang]
        os.chdir(os.path.join(args.output_dir, lang))
        default_processors = {}
        if lang in allowed_empty_languages or lang in no_pretrain_languages:
            pass
        else:
            pos_dependencies = get_pos_dependencies(lang, ud_package)
            depparse_dependencies = get_depparse_dependencies(lang, ud_package)
            pretrains_needed.update([dep['package'] for dep in pos_dependencies if dep['model'] == 'pretrain'])
            pretrains_needed.update([dep['package'] for dep in depparse_dependencies if dep['model'] == 'pretrain'])

        if lang in default_ners:
            ner_package = default_ners[lang]
            ner_dependencies = get_ner_dependencies(lang, ner_package)
            if ner_dependencies is not None:
                pretrains_needed.update([dep['package'] for dep in ner_dependencies if dep['model'] == 'pretrain'])
        if lang in default_charlms:
            charlm_package = default_charlms[lang]
        if lang in default_sentiment:
            sentiment_package = default_sentiment[lang]
            sentiment_dependencies = get_sentiment_dependencies(lang, package)
            pretrains_needed.update([dep['package'] for dep in sentiment_dependencies if dep['model'] == 'pretrain'])
        if lang in default_constituency:
            constituency_package = default_constituency[lang]
            constituency_dependencies = get_con_dependencies(lang, constituency_package)
            pretrains_needed.update([dep['package'] for dep in constituency_dependencies if dep['model'] == 'pretrain'])

        # pretrain doesn't really need to be here, but by putting it here,
        # we preserve any existing default.zip files with no other changes
        # when rebuilding the resources
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

        with zipfile.ZipFile(os.path.join('models', 'default.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for processor in processors:
                if processor == 'pretrain':
                    for package in sorted(pretrains_needed):
                        filename = os.path.join(args.output_dir, lang, "models", processor, package + '.pt')
                        if os.path.exists(filename):
                            print("   Model {} package {}: file {}".format(processor, package, filename))
                            zipf.write(filename=os.path.join("models", processor, package + '.pt'),
                                       arcname=os.path.join(processor, package + '.pt'))
                        else:
                            raise FileNotFoundError(f"Pretrain package {package} needed for {lang} but cannot be found at {filename}")

                    # done specifically with pretrains
                    continue

                if processor == 'ner': package = ner_package
                elif processor in ['forward_charlm', 'backward_charlm']: package = charlm_package
                elif processor == 'sentiment': package = sentiment_package
                elif processor == 'constituency': package = constituency_package
                elif processor == 'langid': package = 'ud' 
                elif processor == 'tokenize' and lang in default_tokenizer: package = default_tokenizer[lang]
                elif processor == 'lemma': package = ud_package + "_nocharlm"
                elif processor == 'pos': package = get_default_pos_package(lang, ud_package)
                elif processor == 'depparse': package = get_default_depparse_package(lang, ud_package)
                else: package = ud_package

                filename = os.path.join(args.output_dir, lang, "models", processor, package + '.pt')

                if os.path.exists(filename):
                    print("   Model {} package {}: file {}".format(processor, package, filename))
                    if processor in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner', 'sentiment', 'constituency', 'langid']:
                        default_processors[processor] = package
                    zipf.write(filename=os.path.join("models", processor, package + '.pt'),
                               arcname=os.path.join(processor, package + '.pt'))
                elif lang in allowed_empty_languages:
                    # we don't have a lot of Thai or Myanmar support yet
                    pass
                elif processor == 'lemma':
                    # a few languages use the identity lemmatizer -
                    # there might be a better way to encode that here
                    default_processors[processor] = "identity"
                    print(" --Model {} package {}: no file {}, assuming identity lemmatizer".format(processor, package, filename))
                elif processor == 'mwt':
                    # some languages don't have MWT, so skip ig
                    # others have pos and depparse built with no pretrain
                    print(" --Model {} package {}: no file {}, skipping".format(processor, package, filename))
                else:
                    raise FileNotFoundError(f"Could not find an expected model file for {lang} {processor} {package} : {filename}")

        default_md5 = get_md5(os.path.join(args.output_dir, lang, 'models', 'default.zip'))
        resources[lang]['default_processors'] = default_processors
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
        elif lang.lower() in three_to_two_letters:
            resources_new[three_to_two_letters[lang.lower()]] = {'alias': lang.lower()}
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
    print("Converting models from %s to %s" % (args.input_dir, args.output_dir))
    process_dirs(args)
    process_defaults(args)
    process_lcode(args)
    process_misc(args)


if __name__ == '__main__':
    main()

