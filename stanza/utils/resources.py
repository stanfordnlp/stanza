"""
utilities for getting resources
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path
import json
import hashlib
import zipfile
import shutil
import logging

from stanza.utils.helper_func import make_table
from stanza.pipeline._constants import TOKENIZE, MWT, POS, LEMMA, DEPPARSE, NER, SENTIMENT, SUPPORTED_TOKENIZERS
from stanza._version import __resources_version__

logger = logging.getLogger('stanza')

# set home dir for default
HOME_DIR = str(Path.home())
DEFAULT_RESOURCES_URL = 'https://raw.githubusercontent.com/stanfordnlp/stanza-resources/master'
DEFAULT_MODEL_DIR = os.getenv('STANZA_RESOURCES_DIR', os.path.join(HOME_DIR, 'stanza_resources'))
PIPELINE_NAMES = [TOKENIZE, MWT, POS, LEMMA, DEPPARSE, NER, SENTIMENT]

# given a language and models path, build a default configuration
def build_default_config(resources, lang, dir, load_list):
    default_config = {}
    for item in load_list:
        processor, package, dependencies = item

        # handle case when spacy or jieba is specified as tokenizer
        if processor == TOKENIZE and package in SUPPORTED_TOKENIZERS:
            default_config[f"{TOKENIZE}_with_{package}"] = True
        # handle case when identity is specified as lemmatizer
        elif processor == LEMMA and package == 'identity':
            default_config[f"{LEMMA}_use_identity"] = True
        else:
            default_config[f"{processor}_model_path"] = os.path.join(dir, lang, processor, package + '.pt')

        if not dependencies: continue
        for dependency in dependencies:
            dep_processor, dep_model = dependency
            default_config[f"{processor}_{dep_processor}_path"] = os.path.join(dir, lang, dep_processor, dep_model + '.pt')

    return default_config

def ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)

def get_md5(path):
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()

def unzip(dir, filename):
    logger.debug(f'Unzip: {dir}/{filename}...')
    with zipfile.ZipFile(os.path.join(dir, filename)) as f:
        f.extractall(dir)

def file_exists(path, md5):
    return os.path.exists(path) and get_md5(path) == md5

def download_file(url, path):
    verbose = logger.level in [0, 10, 20]
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True, disable=not verbose, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))

def request_file(url, path, md5=None):
    ensure_dir(Path(path).parent)
    if file_exists(path, md5):
        logger.info(f'File exists: {path}.')
        return
    download_file(url, path)
    assert(not md5 or file_exists(path, md5))

def sort_processors(processor_list):
    sorted_list = []
    for processor in PIPELINE_NAMES:
        for item in processor_list:
            if item[0] == processor:
                sorted_list.append(item)
    return sorted_list

def maintain_processor_list(resources, lang, package, processors):
    processor_list = {}
    # resolve processor models
    if processors:
        logger.debug(f'Processing parameter "processors"...')
        for key, value in processors.items():
            assert(key in PIPELINE_NAMES)
            assert(isinstance(key, str) and isinstance(value, str))
            # check if keys and values can be found
            if key in resources[lang] and value in resources[lang][key]:
                logger.debug(f'Find {key}: {value}.')
                processor_list[key] = value
            # allow values to be default in some cases
            elif key in resources[lang]['default_processors'] and value == 'default':
                logger.debug(f'Find {key}: {resources[lang]["default_processors"][key]}.')
                processor_list[key] = resources[lang]['default_processors'][key]
            # allow tokenize to be set to "spacy" or "jieba"
            elif key == TOKENIZE and value in SUPPORTED_TOKENIZERS:
                logger.debug(f'Find {key}: {value}. Using external {value} library as tokenizer.')
                processor_list[key] = value
            # allow lemma to be set to "identity"
            elif key == LEMMA and value == 'identity':
                logger.debug(f'Find {key}: {value}. Using identical lemmatizer.')
                processor_list[key] = value
            # cannot find and warn user
            else:
                logger.warning(f'Can not find {key}: {value} from official model list. Ignoring it.')
    # resolve package
    if package:
        logger.debug(f'Processing parameter "package"...')
        if package == 'default':
            for key, value in resources[lang]['default_processors'].items():
                if key not in processor_list:
                    logger.debug(f'Find {key}: {value}.')
                    processor_list[key] = value
        else:
            flag = False
            for key in PIPELINE_NAMES:
                if key not in resources[lang]: continue
                if package in resources[lang][key]:
                    flag = True
                    if key not in processor_list:
                        logger.debug(f'Find {key}: {package}.')
                        processor_list[key] = package
                    else:
                        logger.debug(f'{key}: {package} is overwritten by {key}: {processors[key]}.')
            if not flag: logger.warning((f'Can not find package: {package}.'))
    processor_list = [[key, value] for key, value in processor_list.items()]
    processor_list = sort_processors(processor_list)
    return processor_list

def add_dependencies(resources, lang, processor_list):
    default_dependencies = resources[lang]['default_dependencies']
    for item in processor_list:
        processor, package = item
        dependencies = default_dependencies.get(processor, None)
        # skip dependency checking for special spacy/jieba tokenizer and identity lemmatizer
        if not any([processor == TOKENIZE and package in SUPPORTED_TOKENIZERS, processor == LEMMA and package == 'identity']):
            dependencies = resources[lang][processor][package].get('dependencies', dependencies)
        if dependencies:
            dependencies = [[dependency['model'], dependency['package']] for dependency in dependencies]
        item.append(dependencies)
    return processor_list

def flatten_processor_list(processor_list):
    flattened_processor_list = []
    dependencies_list = []
    for item in processor_list:
        processor, package, dependencies = item
        flattened_processor_list.append([processor, package])
        if dependencies: dependencies_list += [tuple(dependency) for dependency in dependencies]
    dependencies_list = [list(item) for item in set(dependencies_list)]
    for processor, package in dependencies_list:
        logger.debug(f'Find dependency {processor}: {package}.')
    flattened_processor_list += dependencies_list
    return flattened_processor_list

def set_logging_level(logging_level, verbose):
    # Check verbose for easy logging control
    if verbose == False:
        logging_level = 'ERROR'
    elif verbose == True:
        logging_level = 'INFO'

    # Set logging level
    logging_level = logging_level.upper()
    all_levels = ['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL', 'FATAL']
    if logging_level not in all_levels:
        raise Exception(f"Unrecognized logging level for pipeline: {logging_level}. Must be one of {', '.join(all_levels)}.")
    logger.setLevel(logging_level)
    return logging_level

def process_pipeline_parameters(lang, dir, package, processors):
    # Check parameter types and convert values to lower case
    if isinstance(lang, str):
        lang = lang.strip().lower()
    elif lang is not None:
        raise Exception(f"The parameter 'lang' should be str, but got {type(lang).__name__} instead.")

    if isinstance(dir, str):
        dir = dir.strip()
    elif dir is not None:
        raise Exception(f"The parameter 'dir' should be str, but got {type(dir).__name__} instead.")

    if isinstance(package, str):
        package = package.strip().lower()
    elif package is not None:
        raise Exception(f"The parameter 'package' should be str, but got {type(package).__name__} instead.")

    if isinstance(processors, str):
        # Special case: processors is str, compatible with older verson
        processors = {processor.strip().lower(): package for processor in processors.split(',')}
        package = None
    elif isinstance(processors, dict):
        processors = {k.strip().lower(): v.strip().lower() for k, v in processors.items()}
    elif processors is not None:
        raise Exception(f"The parameter 'processors' should be dict or str, but got {type(processors).__name__} instead.")

    return lang, dir, package, processors

# main download function
def download(lang='en', dir=DEFAULT_MODEL_DIR, package='default', processors={}, logging_level='INFO', verbose=None):
    # set global logging level
    set_logging_level(logging_level, verbose)
    # process different pipeline parameters
    lang, dir, package, processors = process_pipeline_parameters(lang, dir, package, processors)

    # Download resources.json to obtain latest packages.
    logger.debug('Downloading resource file...')
    request_file(f'{DEFAULT_RESOURCES_URL}/resources_{__resources_version__}.json', os.path.join(dir, 'resources.json'))
    resources = json.load(open(os.path.join(dir, 'resources.json')))
    if lang not in resources:
        raise Exception(f'Unsupported language: {lang}.')
    if 'alias' in resources[lang]:
        logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
        lang = resources[lang]['alias']
    lang_name = resources[lang]['lang_name'] if 'lang_name' in resources[lang] else ''
    url = resources['url']

    # Default: download zipfile and unzip
    if package == 'default' and (processors is None or len(processors) == 0):
        logger.info(f'Downloading default packages for language: {lang} ({lang_name})...')
        request_file(f'{url}/{__resources_version__}/{lang}/default.zip', os.path.join(dir, lang, f'default.zip'), md5=resources[lang]['default_md5'])
        unzip(os.path.join(dir, lang), 'default.zip')
    # Customize: maintain download list
    else:
        download_list = maintain_processor_list(resources, lang, package, processors)
        download_list = add_dependencies(resources, lang, download_list)
        download_list = flatten_processor_list(download_list)
        download_table = make_table(['Processor', 'Package'], download_list)
        logger.info(f'Downloading these customized packages for language: {lang} ({lang_name})...\n{download_table}')

        # Download packages
        for key, value in download_list:
            try:
                request_file(f'{url}/{__resources_version__}/{lang}/{key}/{value}.pt', os.path.join(dir, lang, key, f'{value}.pt'), md5=resources[lang][key][value]['md5'])
            except KeyError as e:
                raise Exception(f"Cannot find the following processor and model name combination: {key}, {value}. Please check if you have provided the correct model name.") from e
    logger.info(f'Finished downloading models and saved to {dir}.')
