"""
Converts a directory of models organized by type into a directory organized by language.

Also produces the resources.json file.

For example, on the cluster, you can do this:

python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models-1.5.0 --output_dir /u/nlp/software/stanza/models/1.5.0 > resources.out 2>&1
nlprun -a stanza-1.2 -q john "python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models-1.5.0 --output_dir /u/nlp/software/stanza/models/1.5.0" -o resources.out

The work happens in two phases:

  - Planning: the input directories are listed (filenames only, no
    model files are read), and the resources, packages, and
    default.zip contents are all computed in memory.  Every problem
    found - a language missing from default_treebanks, a default
    package with no matching model, a dependency on a model which
    doesn't exist, etc - is collected, and if there are any, they are
    all reported and the script exits before copying anything.
    --check_only stops after this phase.

  - Building: the models are copied, hashed, and zipped, using
    --num_workers threads.

A cache file, .prepare_resources_cache.json, is kept in the output
directory.  It records the stat and md5 of each model and default.zip
written.  On a rerun, a model whose input and output files are both
unchanged is not copied again, and a default.zip whose members are all
unchanged is not rebuilt.  --force ignores the cache.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import hashlib
import shutil
import sys
import threading
import time
import traceback
import zipfile

from stanza import __resources_version__
from stanza.models.common.constant import lcode2lang, two_to_three_letters, three_to_two_letters, extra_lcode_to_lang
from stanza.resources.default_packages import PACKAGES, TRANSFORMERS, TRANSFORMER_NICKNAMES
from stanza.resources.default_packages import *
from stanza.utils.datasets.prepare_lemma_classifier import DATASET_MAPPING as LEMMA_CLASSIFIER_DATASETS
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/u/nlp/software/stanza/models/current-models-%s" % __resources_version__, help='Input dir for various models.  Defaults to the recommended home on the nlp cluster')
    parser.add_argument('--output_dir', type=str, default="/u/nlp/software/stanza/models/%s" % __resources_version__, help='Output dir for various models.')
    parser.add_argument('--packages_only', action='store_true', default=False, help='Only build the package maps instead of rebuilding everything')
    parser.add_argument('--lang', type=str, default=None, help='Only process this language or a comma-separated list of languages.  If left blank, will prepare all languages.  To use this argument, a previous prepared resources with all of the languages is necessary.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads to use when copying, hashing, and zipping models')
    parser.add_argument('--force', action='store_true', default=False, help='Copy every model and rebuild every default.zip, even if the cache says the output is up to date')
    parser.add_argument('--check_only', action='store_true', default=False, help='Only check for problems in the models and default_packages.  Nothing is written')
    args = parser.parse_args()
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.lang is not None:
        args.lang = ",".join(args.lang.strip().split())
    return args


allowed_empty_languages = [
    # only tokenize and NER for Myanmar right now (soon...)
    "my",
    # put the PA model out there even though nothing else is ready
    "pa",
]

# map processor name to file ending
# the order of this dict determines the order in which default.zip files are built
# changing it will necessitate rebuilding all of the default.zip files
# not a disaster, but it would involve a bunch of uploading
processor_to_ending = {
    "tokenize": "tokenizer",
    "mwt": "mwt_expander",
    "lemma": "lemmatizer",
    "pos": "tagger",
    "depparse": "parser",
    "pretrain": "pretrain",
    "ner": "nertagger",
    "forward_charlm": "forward_charlm",
    "backward_charlm": "backward_charlm",
    "sentiment": "sentiment",
    "constituency": "constituency",
    "coref": "coref",
    "langid": "langid",
}
ending_to_processor = {j: i for i, j in processor_to_ending.items()}
PROCESSORS = list(processor_to_ending.keys())

def ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)


COPY_CHUNK_SIZE = 16 * 1024 * 1024

def get_md5(path):
    """
    md5 of a file, read in chunks so that large zips don't have to fit in memory
    """
    md5 = hashlib.md5()
    with open(path, 'rb') as fin:
        while True:
            chunk = fin.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def copy_file_with_md5(src, dst):
    """
    Copy src to dst, including the stat info, and return the md5 of the data

    The data is hashed as it is copied, so each file is only read once.
    The copy is written to a temp file and then renamed, so an
    interrupted copy never leaves a partial file at dst.
    """
    ensure_dir(Path(dst).parent)
    tmp = dst + ".partial"
    md5 = hashlib.md5()
    with open(src, 'rb') as fin, open(tmp, 'wb') as fout:
        while True:
            chunk = fin.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)
            fout.write(chunk)
    shutil.copystat(src, tmp)
    os.replace(tmp, dst)
    return md5.hexdigest()


def stat_signature(path):
    """
    [size, mtime_ns] of the file, or None if it doesn't exist
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return [st.st_size, st.st_mtime_ns]


class OutputCache:
    """
    Remembers the md5 of each file written to the output directory

    Each entry is keyed by the path relative to the output directory.
    An entry is only trusted if the output file (and the input file,
    for models) still has the same size and mtime as when the entry
    was recorded.  The cache is thread safe and is periodically
    flushed to disk, so an interrupted run keeps most of its work.
    """
    CACHE_NAME = ".prepare_resources_cache.json"
    SAVE_EVERY = 25

    def __init__(self, output_dir, use_existing=True):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, self.CACHE_NAME)
        self.entries = {}
        self.lock = threading.Lock()
        self.unsaved = 0
        if use_existing and os.path.exists(self.path):
            try:
                with open(self.path) as fin:
                    self.entries = json.load(fin)
            except (OSError, ValueError) as e:
                print("WARNING: could not read %s, rebuilding everything: %s" % (self.path, e))
                self.entries = {}

    def key(self, path):
        return os.path.relpath(path, self.output_dir)

    def lookup_model(self, src, dst):
        """
        md5 of dst if it is an unchanged copy of an unchanged src, otherwise None
        """
        with self.lock:
            entry = self.entries.get(self.key(dst))
        if not entry or entry.get('src') != src:
            return None
        if entry.get('src_stat') != stat_signature(src) or entry.get('dst_stat') != stat_signature(dst):
            return None
        return entry.get('md5')

    def store_model(self, src, dst, md5):
        self.store(dst, {'src': src, 'src_stat': stat_signature(src), 'dst_stat': stat_signature(dst), 'md5': md5})

    def lookup_zip(self, dst, members):
        """
        md5 of the zip at dst if it was built from exactly these [arcname, md5] members and hasn't changed since
        """
        with self.lock:
            entry = self.entries.get(self.key(dst))
        if not entry or entry.get('members') != members:
            return None
        if entry.get('dst_stat') != stat_signature(dst):
            return None
        return entry.get('md5')

    def store_zip(self, dst, members, md5):
        self.store(dst, {'members': members, 'dst_stat': stat_signature(dst), 'md5': md5})

    def store(self, dst, entry):
        with self.lock:
            self.entries[self.key(dst)] = entry
            self.unsaved += 1
            if self.unsaved >= self.SAVE_EVERY:
                self._save()

    def save(self):
        with self.lock:
            self._save()

    def _save(self):
        ensure_dir(self.output_dir)
        tmp = self.path + ".partial"
        with open(tmp, 'w') as fout:
            json.dump(self.entries, fout, indent=1)
        os.replace(tmp, self.path)
        self.unsaved = 0


def write_resources(resources, output_dir):
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, 'resources.json'), 'w') as fout:
        json.dump(resources, fout, indent=2)


def load_resources(output_dir):
    with open(os.path.join(output_dir, 'resources.json')) as fin:
        return json.load(fin)


def run_in_threads(func, jobs, num_workers, desc):
    """
    Run func(*job) for each job, returning the results in job order

    If any job fails, the jobs which haven't started yet are cancelled
    and the exception is raised.
    """
    results = [None] * len(jobs)
    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = {executor.submit(func, *job): idx for idx, job in enumerate(jobs)}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                results[futures[future]] = future.result()
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    return results


def describe_error(context, e):
    """
    Format an exception caught while checking for problems

    Expected errors (bad config, missing models) are summarized in one
    line.  Anything else is probably a bug, so it gets the full traceback.
    """
    if isinstance(e, (AssertionError, RuntimeError, FileNotFoundError, ValueError)):
        return "%s: %s" % (context, e)
    return "%s: %s: %s\n%s" % (context, type(e).__name__, e, traceback.format_exc())


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

def split_package(package, default_use_charlm=True):
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
    underscore = package.rfind("_")
    if underscore >= 0:
        # +1 to skip the underscore
        nickname = package[underscore+1:]
        if nickname in known_nicknames():
            return package[:underscore], True, True

    # guess it was a model which wasn't built with the new naming convention of putting the pretrain type at the end
    # assume WV and charlm... if the language / package doesn't allow for one, that should be caught later
    return package, True, default_use_charlm

def get_pretrain_package(lang, package, model_pretrains, default_pretrains):
    package, uses_pretrain, _ = split_package(package)

    if not uses_pretrain or lang in no_pretrain_languages:
        return None
    elif model_pretrains is not None and lang in model_pretrains and package in model_pretrains[lang]:
        return model_pretrains[lang][package]
    elif lang in default_pretrains:
        return default_pretrains[lang]

    raise RuntimeError("pretrain not specified for lang %s package %s" % (lang, package))

def get_charlm_package(lang, package, model_charlms, default_charlms, default_use_charlm=True):
    package, _, uses_charlm = split_package(package, default_use_charlm)

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

def get_lemma_pretrain_package(lang, package):
    package, uses_pretrain, uses_charlm = split_package(package)
    if not uses_pretrain:
        return None
    if not uses_charlm:
        # currently the contextual lemma classifier is only active
        # for the charlm lemmatizers
        return None
    if "%s_%s" % (lang, package) not in LEMMA_CLASSIFIER_DATASETS:
        return None
    return get_pretrain_package(lang, package, {}, default_pretrains)

def get_lemma_charlm_package(lang, package):
    return get_charlm_package(lang, package, lemma_charlms, default_charlms)

def get_lemma_dependencies(lang, package):
    dependencies = []

    pretrain_package = get_lemma_pretrain_package(lang, package)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})

    charlm_package = get_lemma_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})

    return dependencies


def get_tokenizer_charlm_package(lang, package):
    return get_charlm_package(lang, package, tokenizer_charlms, default_charlms, default_use_charlm=False)

def get_tokenizer_dependencies(lang, package):
    dependencies = []
    charlm_package = get_tokenizer_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
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

def get_ner_pretrain_package(lang, package):
    return get_pretrain_package(lang, package, ner_pretrains, default_pretrains)

def get_ner_dependencies(lang, package):
    dependencies = []

    pretrain_package = get_ner_pretrain_package(lang, package)
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

def get_dependencies(processor, lang, package):
    """
    Get the dependencies for a particular lang/package based on the package name

    The package can include descriptors such as _nopretrain, _nocharlm, _charlm
    which inform whether or not this particular model uses charlm or pretrain
    """
    if processor == 'depparse':
        return get_depparse_dependencies(lang, package)
    elif processor == 'lemma':
        return get_lemma_dependencies(lang, package)
    elif processor == 'pos':
        return get_pos_dependencies(lang, package)
    elif processor == 'ner':
        return get_ner_dependencies(lang, package)
    elif processor == 'sentiment':
        return get_sentiment_dependencies(lang, package)
    elif processor == 'constituency':
        return get_con_dependencies(lang, package)
    elif processor == 'tokenize':
        return get_tokenizer_dependencies(lang, package)
    return {}

def selected_langs(args):
    return args.lang.split(",") if args.lang else None

def scan_input_dirs(args, errors):
    """
    Build resources from the filenames in the input directories

    No model files are read or copied here.  Each model gets an entry
    with its dependencies and a placeholder md5, which is filled in by
    copy_models.

    Returns resources and a list of (input_path, output_path, lang, processor, package) to copy
    """
    langs = selected_langs(args)
    resources = {}
    if langs:
        resources = load_resources(args.output_dir)
        # the selected languages get rebuilt from scratch
        # otherwise, any models which were deleted would still be in the resources
        for lang in langs:
            resources[lang] = {}

    copies = []
    for model_dir in sorted(os.listdir(args.input_dir)):
        dir_path = os.path.join(args.input_dir, model_dir)
        if not os.path.isdir(dir_path):
            continue
        for model in sorted(os.listdir(dir_path)):
            if not model.endswith('.pt'): continue
            try:
                lang, package, processor = split_model_name(model)
            except (AssertionError, ValueError) as e:
                errors.append("%s/%s: cannot parse model name: %s" % (model_dir, model, e))
                continue
            if langs and lang not in langs:
                continue

            try:
                dependencies = get_dependencies(processor, lang, package)
            except Exception as e:
                errors.append(describe_error("%s/%s" % (model_dir, model), e))
                dependencies = None

            # md5 is filled in when the model is copied
            # it is put in first so that the key order matches in resources.json
            entry = {'md5': None}
            if dependencies:
                entry['dependencies'] = dependencies
            resources.setdefault(lang, {}).setdefault(processor, {})[package] = entry

            input_path = os.path.join(dir_path, model)
            output_path = os.path.join(args.output_dir, lang, "models", processor, package + '.pt')
            copies.append((input_path, output_path, lang, processor, package))
    print("Found %d models in %s" % (len(copies), args.input_dir))
    return resources, copies

def check_dependencies(resources, copies, errors):
    """
    Check that every dependency of every scanned model is itself a known model
    """
    for _, _, lang, processor, package in copies:
        entry = resources[lang][processor][package]
        for dependency in entry.get('dependencies', []):
            dep_model, dep_package = dependency['model'], dependency['package']
            if dep_package is None or dep_package not in resources[lang].get(dep_model, {}):
                errors.append("%s %s %s depends on %s %s, which does not exist" % (lang, processor, package, dep_model, dep_package))

def copy_models(resources, copies, cache, num_workers):
    """
    Copy each model to the output directory, filling in its md5 in resources

    Models which the cache says are already up to date are not copied.
    """
    def copy_one(input_path, output_path, lang, processor, package):
        md5 = cache.lookup_model(input_path, output_path)
        if md5 is not None:
            return md5, False
        md5 = copy_file_with_md5(input_path, output_path)
        cache.store_model(input_path, output_path, md5)
        return md5, True

    try:
        results = run_in_threads(copy_one, copies, num_workers, "Copying models")
    finally:
        cache.save()
    for (_, _, lang, processor, package), (md5, _) in zip(copies, results):
        resources[lang][processor][package]['md5'] = md5
    num_copied = sum(1 for _, copied in results if copied)
    print("Copied %d models, %d were already up to date" % (num_copied, len(copies) - num_copied))

def get_default_pos_package(lang, ud_package, known_resources):
    charlm_package = get_pos_charlm_package(lang, ud_package)
    if charlm_package is not None:
        charlm_package = ud_package + "_charlm"
        nocharlm_package = ud_package + "_nocharlm"
        if charlm_package in known_resources.get("pos", {}):
            return charlm_package
        else:
            return nocharlm_package
    if lang in no_pretrain_languages:
        return ud_package + "_nopretrain"
    transformer = TRANSFORMER_NICKNAMES.get(TRANSFORMERS.get(lang, None), None)
    transformer_package = "%s_%s" % (ud_package, transformer)
    nocharlm_package = "%s_nocharlm" % ud_package
    # TODO: use a defaultdict here instead
    if nocharlm_package in known_resources.get("pos", {}):
        return nocharlm_package
    if transformer_package in known_resources.get("pos", {}):
        return transformer_package
    # this will probably cause a problem when there is no model of this name
    return ud_package + "_nocharlm"

def get_default_depparse_package(lang, ud_package, known_resources):
    charlm_package = get_depparse_charlm_package(lang, ud_package)
    if charlm_package is not None:
        charlm_package = ud_package + "_charlm"
        nocharlm_package = ud_package + "_nocharlm"
        if charlm_package in known_resources.get("depparse", {}):
            return charlm_package
        else:
            return nocharlm_package
    if lang in no_pretrain_languages:
        return ud_package + "_nopretrain"
    transformer = TRANSFORMER_NICKNAMES.get(TRANSFORMERS.get(lang, None), None)
    transformer_package = "%s_%s" % (ud_package, transformer)
    nocharlm_package = "%s_nocharlm" % ud_package
    # TODO: use a defaultdict here instead
    if nocharlm_package in known_resources.get("depparse", {}):
        return nocharlm_package
    if transformer_package in known_resources.get("depparse", {}):
        return transformer_package
    # this will probably cause a problem when there is no model of this name
    return ud_package + "_nocharlm"

def is_packaged_language(resources, lang):
    """
    Whether this entry in resources is a language which gets packages and a default.zip

    url, alias, and lang_name are checked in case we are rerunning on an already built resources.json
    """
    if lang == 'url':
        return False
    if 'alias' in resources[lang]:
        return False
    if all(k in ("backward_charlm", "forward_charlm", "pretrain", "lang_name") for k in resources[lang].keys()):
        return False
    if lang in allowed_empty_languages and lang not in default_treebanks:
        return False
    return True

def plan_default_zips(resources, args, errors):
    """
    Figure out which models go in each language's default.zip

    Every model needed by the default package, or by one of its
    dependencies, must be in resources.  Missing models are added to errors.

    Returns a list of (lang, zip_path, [(filename, processor, package), ...])
    """
    langs = selected_langs(args)
    zip_plans = []
    for lang in resources:
        if not is_packaged_language(resources, lang):
            continue
        if lang not in default_treebanks:
            # already reported by build_packages
            continue
        if langs and lang not in langs:
            continue
        if PACKAGES not in resources[lang]:
            # build_packages failed for this language and already reported why
            continue

        models_needed = defaultdict(set)
        try:
            packages = resources[lang][PACKAGES]["default"]
            for processor, package in packages.items():
                if processor == 'lemma' and package == 'identity':
                    continue
                if processor == 'optional':
                    continue
                models_needed[processor].add(package)
                dependencies = get_dependencies(processor, lang, package)
                for dependency in dependencies:
                    models_needed[dependency['model']].add(dependency['package'])
        except Exception as e:
            errors.append(describe_error("%s default.zip" % lang, e))
            continue

        model_files = []
        for processor in PROCESSORS:
            if processor in models_needed:
                for package in sorted(models_needed[processor], key=str):
                    if package not in resources[lang].get(processor, {}):
                        errors.append("Processor %s package %s needed for %s default.zip, but there is no such model" % (processor, package, lang))
                        continue
                    filename = os.path.join(args.output_dir, lang, "models", processor, package + '.pt')
                    model_files.append((filename, processor, package))

        zip_path = os.path.join(args.output_dir, lang, 'models', 'default.zip')
        zip_plans.append((lang, zip_path, model_files))
    return zip_plans

def build_default_zips(resources, zip_plans, cache, num_workers):
    """
    Write each planned default.zip and record its md5 in resources

    A zip whose members are the same models (by md5) as the cached
    version is not rebuilt.  The member order follows PROCESSORS, so
    the zips come out the same as long as the models do.
    """
    def build_one(lang, zip_path, model_files):
        members = [[os.path.join(processor, package + '.pt'), resources[lang][processor][package]['md5']]
                   for _, processor, package in model_files]
        md5 = cache.lookup_zip(zip_path, members)
        if md5 is not None:
            return md5, False
        tmp = zip_path + ".partial"
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename, processor, package in model_files:
                zipf.write(filename=filename, arcname=os.path.join(processor, package + '.pt'))
        os.replace(tmp, zip_path)
        md5 = get_md5(zip_path)
        cache.store_zip(zip_path, members, md5)
        return md5, True

    for lang, _, model_files in zip_plans:
        print('Default models for language %s' % lang)
        for filename, processor, package in model_files:
            print("   Model {} package {}: file {}".format(processor, package, filename))

    try:
        results = run_in_threads(build_one, zip_plans, num_workers, "Building default.zip")
    finally:
        cache.save()
    for (lang, _, _), (md5, _) in zip(zip_plans, results):
        resources[lang]['default_md5'] = md5
    num_built = sum(1 for _, built in results if built)
    print("Built %d default.zip files, %d were already up to date" % (num_built, len(zip_plans) - num_built))

def get_default_processors(resources, lang):
    """
    Build a default package for this language

    Will add each of pos, lemma, depparse, etc if those are available
    Uses the existing models scraped from the language directories into resources.json, as relevant
    """
    if lang == "multilingual":
        return {"langid": "ud"}

    default_package = default_treebanks[lang]
    default_processors = {}
    if 'tokenize' not in resources[lang]:
        raise AssertionError("No tokenizer models found for %s" % lang)
    if lang in default_tokenizer:
        default_processors['tokenize'] = default_tokenizer[lang]
    else:
        tokenize_package = default_package
        if tokenize_package not in resources[lang]['tokenize']:
            tokenize_package = default_package + "_nocharlm"
        if tokenize_package not in resources[lang]['tokenize']:
            tokenize_package = default_package + "_charlm"
            if tokenize_package in resources[lang]['tokenize']:
                print("WARNING: nocharlm tokenizer for %s model does not exist, but %s does" % (default_package, tokenize_package))
        if tokenize_package not in resources[lang]['tokenize']:
            raise AssertionError("Can't find a tokenizer package for %s!  Tried %s, %s, %s" % (lang, default_package, (default_package + "_nocharlm"), tokenize_package))
        default_processors['tokenize'] = tokenize_package

    if 'mwt' in resources[lang] and default_package in resources[lang]['mwt']:
        # if this doesn't happen, we just skip MWT
        default_processors['mwt'] = default_package

    if 'lemma' in resources[lang]:
        expected_lemma = default_package + "_nocharlm"
        if expected_lemma in resources[lang]['lemma']:
            default_processors['lemma'] = expected_lemma
        else:
            expected_lemma = default_package + "_charlm"
            if expected_lemma in resources[lang]['lemma']:
                default_processors['lemma'] = expected_lemma
                print("WARNING: nocharlm lemmatizer for %s model does not exist, but %s does" % (default_package, expected_lemma))
    elif lang not in allowed_empty_languages:
        default_processors['lemma'] = 'identity'

    if 'pos' in resources[lang]:
        default_processors['pos'] = get_default_pos_package(lang, default_package, resources[lang])
        if default_processors['pos'] not in resources[lang]['pos']:
            raise AssertionError("Expected POS model not in resources: %s" % default_processors['pos'])
    elif lang not in allowed_empty_languages:
        raise AssertionError("Expected to find POS models for language %s" % lang)

    if 'depparse' in resources[lang]:
        default_processors['depparse'] = get_default_depparse_package(lang, default_package, resources[lang])
        if default_processors['depparse'] not in resources[lang]['depparse']:
            raise AssertionError("Expected depparse model not in resources: %s" % default_processors['depparse'])
    elif lang not in allowed_empty_languages:
        raise AssertionError("Expected to find depparse models for language %s" % lang)

    if lang in default_ners:
        default_processors['ner'] = default_ners[lang]

    if lang in default_sentiment:
        default_processors['sentiment'] = default_sentiment[lang]

    if lang in default_constituency:
        default_processors['constituency'] = default_constituency[lang]

    optional = get_default_optional_processors(resources, lang)
    if optional:
        default_processors['optional'] = optional

    return default_processors

def get_default_optional_processors(resources, lang):
    optional_processors = {}
    if lang in optional_constituency:
        optional_processors['constituency'] = optional_constituency[lang]

    if lang in optional_coref:
        optional_processors['coref'] = optional_coref[lang]

    return optional_processors

def update_processor_add_transformer(resources, lang, current_processors, processor, transformer):
    if processor not in current_processors:
        return

    new_model = current_processors[processor].replace('_charlm', "_" + transformer).replace('_nocharlm', "_" + transformer)
    if new_model in resources[lang][processor]:
        current_processors[processor] = new_model
    else:
        print("WARNING: wanted to use %s for %s accurate %s, but that model does not exist" % (new_model, lang, processor))

def get_default_accurate(resources, lang):
    """
    A package that, if available, uses charlm and transformer models for each processor
    """
    default_processors = get_default_processors(resources, lang)

    tokenizer_model = default_processors['tokenize']
    if tokenizer_model.endswith('_nocharlm'):
        tokenizer_model = tokenizer_model.replace('_nocharlm', '_charlm')
    elif 'charlm' not in tokenizer_model:
        tokenizer_model = tokenizer_model + '_charlm'
    if tokenizer_model.endswith('_charlm') and tokenizer_model in resources[lang]['tokenize']:
        default_processors['tokenize'] = tokenizer_model
        print("TOKENIZE found a charlm version %s for %s default_accurate" % (tokenizer_model, lang))

    if 'lemma' in default_processors and default_processors['lemma'] != 'identity':
        lemma_model = default_processors['lemma']
        lemma_model = lemma_model.replace('_nocharlm', '_charlm')
        charlm_package = get_lemma_charlm_package(lang, lemma_model)
        if charlm_package is not None:
            if lemma_model in resources[lang]['lemma']:
                default_processors['lemma'] = lemma_model
            else:
                print("WARNING: wanted to use %s for %s default_accurate lemma, but that model does not exist" % (lemma_model, lang))

    transformer = TRANSFORMER_NICKNAMES.get(TRANSFORMERS.get(lang, None), None)
    if transformer is not None:
        for processor in ('pos', 'depparse', 'constituency', 'sentiment'):
            update_processor_add_transformer(resources, lang, default_processors, processor, transformer)
        if 'ner' in default_processors and (default_processors['ner'].endswith("_charlm") or default_processors['ner'].endswith("_nocharlm")):
            update_processor_add_transformer(resources, lang, default_processors, "ner", transformer)

    optional = get_optional_accurate(resources, lang)
    if optional:
        default_processors['optional'] = optional

    return default_processors

def get_optional_accurate(resources, lang):
    optional_processors = get_default_optional_processors(resources, lang)

    transformer = TRANSFORMER_NICKNAMES.get(TRANSFORMERS.get(lang, None), None)
    if transformer is not None:
        for processor in ('pos', 'depparse', 'constituency', 'sentiment'):
            update_processor_add_transformer(resources, lang, optional_processors, processor, transformer)

    if lang in optional_coref:
        optional_processors['coref'] = optional_coref[lang]

    return optional_processors


def get_default_fast(resources, lang):
    """
    Build a packages entry which only has the nocharlm models

    Will make it easy for people to use the lower tier of models

    We do this by building the same default package as normal,
    then switching everything out for the lower tier model when possible.
    We also remove constituency, as it is super slow.
    Note that in the case of a language which doesn't have a charlm,
    that means we wind up building the same for default and default_nocharlm
    """
    default_processors = get_default_processors(resources, lang)

    # this is a slow model and we don't have non-charlm versions of it yet
    if 'constituency' in default_processors:
        default_processors.pop('constituency')

    for processor, model in default_processors.items():
        if "_charlm" in model:
            nocharlm = model.replace("_charlm", "_nocharlm")
            if nocharlm not in resources[lang][processor]:
                print("WARNING: wanted to use %s for %s default_fast processor %s, but that model does not exist" % (nocharlm, lang, processor))
            else:
                default_processors[processor] = nocharlm

    return default_processors

def build_lang_packages(resources, lang):
    """
    Build a package for a language's default processors and all of the treebanks specifically used for that language
    """
    default_processors = get_default_processors(resources, lang)

    # build the packages in a separate dict so that a failure partway
    # through doesn't leave a half built PACKAGES entry
    packages = {}
    packages['default'] = default_processors

    if lang not in no_pretrain_languages and lang != "multilingual":
        packages['default_fast'] = get_default_fast(resources, lang)
        packages['default_accurate'] = get_default_accurate(resources, lang)

    # Now we loop over each of the tokenizers for this language
    # ... we use this as a proxy for the available UD treebanks
    # This loop also catches things such as "craft" which are
    # included treebanks that aren't UD
    # We then create a package in the packages dict for each of those treebanks
    if 'tokenize' in resources[lang]:
        for package in resources[lang]['tokenize']:
            package, _, _ = split_package(package)
            if package in packages:
                # can happen in the case of a _nocharlm and _charlm version of the tokenizer
                continue

            processors = {}
            # TODO: when we rebuild all the models, make all the tokenizers say _nocharlm
            if package in resources[lang]['tokenize']:
                processors["tokenize"] = package
            elif package + "_nocharlm" in resources[lang]['tokenize']:
                processors["tokenize"] = package + "_nocharlm"
            elif package + "_charlm" in resources[lang]['tokenize']:
                processors["tokenize"] = package + "_charlm"
            else:
                raise AssertionError("Should have found a tokenizer for lang %s package %s" % (lang, package))

            if "mwt" in resources[lang] and package in resources[lang]["mwt"]:
                processors["mwt"] = package

            if "pos" in resources[lang]:
                if package + "_charlm" in resources[lang]["pos"]:
                    processors["pos"] = package + "_charlm"
                elif package + "_nocharlm" in resources[lang]["pos"]:
                    processors["pos"] = package + "_nocharlm"

            if "lemma" in resources[lang] and "pos" in processors:
                lemma_package = package + "_nocharlm"
                if lemma_package in resources[lang]["lemma"]:
                    processors["lemma"] = lemma_package
                else:
                    lemma_package = package + "_charlm"
                    if lemma_package in resources[lang]['lemma']:
                        processors['lemma'] = lemma_package
                        print("WARNING: nocharlm lemmatizer for %s model does not exist, but %s does" % (package, lemma_package))

            if "depparse" in resources[lang] and "pos" in processors:
                depparse_package = None
                if package + "_charlm" in resources[lang]["depparse"]:
                    depparse_package = package + "_charlm"
                elif package + "_nocharlm" in resources[lang]["depparse"]:
                    depparse_package = package + "_nocharlm"
                # we want to set the lemma first if it's identity
                # THEN set the depparse
                if depparse_package is not None:
                    if "lemma" not in processors:
                        processors["lemma"] = "identity"
                    processors["depparse"] = depparse_package

            packages[package] = processors

    # TODO: eventually we can remove default_processors
    # For now, we want to keep this so that v1.5.1 is compatible
    # with the next iteration of resources files
    resources[lang]['default_processors'] = default_processors
    resources[lang][PACKAGES] = packages

def build_packages(resources, args, errors):
    """
    Build the packages for each language, adding any problems found to errors
    """
    langs = selected_langs(args)
    for lang in resources:
        if not is_packaged_language(resources, lang):
            continue
        if lang not in default_treebanks:
            errors.append(f'{lang} not in default treebanks!!!')
            continue

        if langs and lang not in langs:
            continue

        try:
            build_lang_packages(resources, lang)
        except Exception as e:
            errors.append(describe_error("%s packages" % lang, e))

def check_packages(resources, args, errors):
    """
    Check that every model named in every package (not just default) exists
    """
    langs = selected_langs(args)
    for lang in resources:
        if langs and lang not in langs:
            continue
        if not is_packaged_language(resources, lang) or PACKAGES not in resources[lang]:
            continue
        for package_name, processors in resources[lang][PACKAGES].items():
            to_check = [(k, v) for k, v in processors.items() if k != 'optional']
            to_check.extend(processors.get('optional', {}).items())
            for processor, package in to_check:
                if processor == 'lemma' and package == 'identity':
                    continue
                if package not in resources[lang].get(processor, {}):
                    errors.append("%s package %s uses %s %s, but there is no such model" % (lang, package_name, processor, package))

def check_lcode(resources, args, errors):
    """
    Check for problems which process_lcode would otherwise hit after all the copying
    """
    if 'multilingual' not in resources:
        errors.append("No multilingual models found.  Is the langid model missing?")
    langs = selected_langs(args)
    for lang in resources:
        if langs and lang not in langs:
            continue
        if lang in ('url', 'multilingual') or 'alias' in resources[lang]:
            continue
        if lang not in lcode2lang:
            errors.append("%s not found in lcode2lang!  It would be left out of resources.json" % lang)

def process_lcode(resources):
    resources_new = {}
    resources_new["multilingual"] = resources["multilingual"]
    for lang in sorted(resources):
        if lang == 'multilingual':
            continue
        if 'alias' in resources[lang]:
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
        for alternative in extra_lcode_to_lang[lang.lower()]:
            if alternative.lower() not in resources_new:
                resources_new[alternative.lower()] = {'alias': lang.lower()}
    print("Processed lcode aliases.  Writing resources.json")
    return resources_new


def process_misc(resources):
    resources['no'] = {'alias': 'nb'}
    resources['zh'] = {'alias': 'zh-hans'}
    # This is intended to be unformatted.  expand_model_url in common.py will fill in the raw string
    # with the appropriate values in order to find the needed model file on huggingface
    resources['url'] = 'https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/v{resources_version}/models/{filename}'
    print("Finalized misc attributes")
    return resources


def report_errors(errors):
    if not errors:
        print("No problems found")
        return
    print()
    print("Found %d problem(s).  Nothing has been copied or written." % len(errors))
    for error in errors:
        print("  " + error)
    sys.exit(1)


def main():
    args = parse_args()
    start = time.time()
    print("Converting models from %s to %s" % (args.input_dir, args.output_dir))

    # Planning: everything here works from filenames and default_packages only,
    # so all problems are found before any time is spent copying
    errors = []
    if args.packages_only:
        resources = load_resources(args.output_dir)
        copies = []
    else:
        resources, copies = scan_input_dirs(args, errors)
        check_dependencies(resources, copies, errors)
    build_packages(resources, args, errors)
    check_packages(resources, args, errors)
    zip_plans = plan_default_zips(resources, args, errors)
    if not args.packages_only:
        check_lcode(resources, args, errors)
    report_errors(errors)
    print("Planning took %.1fs" % (time.time() - start))
    if args.check_only:
        return

    if args.packages_only:
        write_resources(resources, args.output_dir)
        print("Wrote packages to resources.json")
        return

    # Building
    cache = OutputCache(args.output_dir, use_existing=not args.force)
    copy_models(resources, copies, cache, args.num_workers)
    print("Copied models.  Writing preliminary resources.json  (%.1fs elapsed)" % (time.time() - start))
    write_resources(resources, args.output_dir)

    build_default_zips(resources, zip_plans, cache, args.num_workers)
    print("Built default zips  (%.1fs elapsed)" % (time.time() - start))

    resources = process_lcode(resources)
    resources = process_misc(resources)
    write_resources(resources, args.output_dir)
    print("Wrote resources.json  (%.1fs total)" % (time.time() - start))


if __name__ == '__main__':
    main()
