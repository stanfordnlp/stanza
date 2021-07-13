"""
Common utilities for Stanza resources.
"""

import os
import requests
from tqdm.auto import tqdm
from pathlib import Path
import json
import hashlib
import zipfile
import shutil
import logging

from stanza.utils.helper_func import make_table
from stanza.pipeline._constants import TOKENIZE, MWT, POS, LEMMA, DEPPARSE, \
    NER, SENTIMENT
from stanza.pipeline.registry import PIPELINE_NAMES, PROCESSOR_VARIANTS
from stanza._version import __resources_version__

logger = logging.getLogger('stanza')

# set home dir for default
HOME_DIR = str(Path.home())
STANFORDNLP_RESOURCES_URL = 'https://nlp.stanford.edu/software/stanza/stanza-resources/'
STANZA_RESOURCES_GITHUB = 'https://raw.githubusercontent.com/stanfordnlp/stanza-resources/'
DEFAULT_RESOURCES_URL = os.getenv('STANZA_RESOURCES_URL', STANZA_RESOURCES_GITHUB + 'main')
DEFAULT_RESOURCES_VERSION = os.getenv(
    'STANZA_RESOURCES_VERSION',
    __resources_version__
)
DEFAULT_MODEL_URL = os.getenv('STANZA_MODEL_URL', 'default')
DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(HOME_DIR, 'stanza_resources')
)

class UnknownProcessorError(ValueError):
    def __init__(self, unknown):
        super().__init__(f"Unknown processor type requested: {unknown}")
        self.unknown_processor = unknown

# given a language and models path, build a default configuration
def build_default_config(resources, lang, model_dir, load_list):
    default_config = {}
    for item in load_list:
        processor, package, dependencies = item

        # handle case when processor variants are used
        if package in PROCESSOR_VARIANTS[processor]:
            default_config[f"{processor}_with_{package}"] = True
        # handle case when identity is specified as lemmatizer
        elif processor == LEMMA and package == 'identity':
            default_config[f"{LEMMA}_use_identity"] = True
        else:
            default_config[f"{processor}_model_path"] = os.path.join(
                model_dir, lang, processor, package + '.pt'
            )

        if not dependencies: continue
        for dependency in dependencies:
            dep_processor, dep_model = dependency
            default_config[f"{processor}_{dep_processor}_path"] = os.path.join(
                model_dir, lang, dep_processor, dep_model + '.pt'
            )

    return default_config

def ensure_dir(path):
    """
    Create dir in case it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_md5(path):
    """
    Get the MD5 value of a path.
    """
    with open(path, 'rb') as fin:
        data = fin.read()
    return hashlib.md5(data).hexdigest()

def unzip(path, filename):
    """
    Fully unzip a file `filename` that's in a directory `dir`.
    """
    logger.debug(f'Unzip: {path}/{filename}...')
    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        f.extractall(path)

def get_root_from_zipfile(filename):
    """
    Get the root directory from a archived zip file.
    """
    zf = zipfile.ZipFile(filename, "r")
    assert len(zf.filelist) > 0, \
        f"Zip file at f{filename} seems to be corrupted. Please check it."
    return os.path.dirname(zf.filelist[0].filename)

def file_exists(path, md5):
    """
    Check if the file at `path` exists and match the provided md5 value.
    """
    return os.path.exists(path) and get_md5(path) == md5

def assert_file_exists(path, md5=None):
    assert os.path.exists(path), "Could not find file at %s" % path
    if md5:
        file_md5 = get_md5(path)
        assert file_md5 == md5, "md5 for %s is %s, expected %s" % (path, file_md5, md5)

def download_file(url, path, proxies, raise_for_status=False):
    """
    Download a URL into a file as specified by `path`.
    """
    verbose = logger.level in [0, 10, 20]
    r = requests.get(url, stream=True, proxies=proxies)
    with open(path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True, \
            disable=not verbose, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))
    if raise_for_status:
        r.raise_for_status()
    return r.status_code

def request_file(url, path, proxies=None, md5=None, raise_for_status=False):
    """
    A complete wrapper over download_file() that also make sure the directory of
    `path` exists, and that a file matching the md5 value does not exist.
    """
    ensure_dir(Path(path).parent)
    if file_exists(path, md5):
        logger.info(f'File exists: {path}.')
        return
    download_file(url, path, proxies, raise_for_status)
    assert_file_exists(path, md5)

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
            assert(isinstance(key, str) and isinstance(value, str))
            if key not in PIPELINE_NAMES:
                raise UnknownProcessorError(key)
            # check if keys and values can be found
            if key in resources[lang] and value in resources[lang][key]:
                logger.debug(f'Found {key}: {value}.')
                processor_list[key] = value
            # allow values to be default in some cases
            elif key in resources[lang]['default_processors'] and value == 'default':
                logger.debug(
                    f'Found {key}: {resources[lang]["default_processors"][key]}.'
                )
                processor_list[key] = resources[lang]['default_processors'][key]
            # allow processors to be set to variants that we didn't implement
            elif value in PROCESSOR_VARIANTS[key]:
                logger.debug(
                    f'Found {key}: {value}. '
                    f'Using external {value} variant for the {key} processor.'
                )
                processor_list[key] = value
            # allow lemma to be set to "identity"
            elif key == LEMMA and value == 'identity':
                logger.debug(
                    f'Found {key}: {value}. Using identity lemmatizer.'
                )
                processor_list[key] = value
            # not a processor in the officially supported processor list
            elif key not in resources[lang]:
                logger.debug(
                    f'{key}: {value} is not officially supported by Stanza, '
                    f'loading it anyway.'
                )
                processor_list[key] = value
            # cannot find the package for a processor and warn user
            else:
                logger.warning(
                    f'Can not find {key}: {value} from official model list. '
                    f'Ignoring it.'
                )
    # resolve package
    if package:
        logger.debug(f'Processing parameter "package"...')
        if package == 'default':
            for key, value in resources[lang]['default_processors'].items():
                if key not in processor_list:
                    logger.debug(f'Found {key}: {value}.')
                    processor_list[key] = value
        else:
            flag = False
            for key in PIPELINE_NAMES:
                if key not in resources[lang]: continue
                if package in resources[lang][key]:
                    flag = True
                    if key not in processor_list:
                        logger.debug(f'Found {key}: {package}.')
                        processor_list[key] = package
                    else:
                        logger.debug(
                            f'{key}: {package} is overwritten by '
                            f'{key}: {processors[key]}.'
                        )
            if not flag: logger.warning((f'Can not find package: {package}.'))
    processor_list = [[key, value] for key, value in processor_list.items()]
    processor_list = sort_processors(processor_list)
    return processor_list

def add_dependencies(resources, lang, processor_list):
    default_dependencies = resources[lang]['default_dependencies']
    for item in processor_list:
        processor, package = item
        dependencies = default_dependencies.get(processor, None)
        # skip dependency checking for external variants of processors and identity lemmatizer
        if not any([
                package in PROCESSOR_VARIANTS[processor],
                processor == LEMMA and package == 'identity'
            ]):
            dependencies = resources[lang].get(processor, {}).get(package, {}) \
                .get('dependencies', dependencies)
        if dependencies:
            dependencies = [[dependency['model'], dependency['package']] \
                for dependency in dependencies]
        item.append(dependencies)
    return processor_list

def flatten_processor_list(processor_list):
    flattened_processor_list = []
    dependencies_list = []
    for item in processor_list:
        processor, package, dependencies = item
        flattened_processor_list.append([processor, package])
        if dependencies:
            dependencies_list += [tuple(dependency) for dependency in dependencies]
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

    if logging_level is None:
        # default logging level of INFO is set in stanza.__init__
        # but the user may have set it via the logging API
        # it should NOT be 0, but let's check to be sure...
        if logger.level == 0:
            logger.setLevel('INFO')
        return logger.level

    # Set logging level
    logging_level = logging_level.upper()
    all_levels = ['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL', 'FATAL']
    if logging_level not in all_levels:
        raise ValueError(
            f"Unrecognized logging level for pipeline: "
            f"{logging_level}. Must be one of {', '.join(all_levels)}."
        )
    logger.setLevel(logging_level)
    return logger.level

def process_pipeline_parameters(lang, model_dir, package, processors):
    # Check parameter types and convert values to lower case
    if isinstance(lang, str):
        lang = lang.strip().lower()
    elif lang is not None:
        raise TypeError(
            f"The parameter 'lang' should be str, "
            f"but got {type(lang).__name__} instead."
        )

    if isinstance(model_dir, str):
        model_dir = model_dir.strip()
    elif model_dir is not None:
        raise TypeError(
            f"The parameter 'model_dir' should be str, "
            f"but got {type(model_dir).__name__} instead."
        )

    if isinstance(package, str):
        package = package.strip().lower()
    elif package is not None:
        raise TypeError(
            f"The parameter 'package' should be str, "
            f"but got {type(package).__name__} instead."
        )

    if isinstance(processors, str):
        # Special case: processors is str, compatible with older version
        processors = {
            processor.strip().lower(): package \
                for processor in processors.split(',')
        }
        package = None
    elif isinstance(processors, dict):
        processors = {
            k.strip().lower(): v.strip().lower() \
                for k, v in processors.items()
        }
    elif processors is not None:
        raise TypeError(
            f"The parameter 'processors' should be dict or str, "
            f"but got {type(processors).__name__} instead."
        )

    return lang, model_dir, package, processors

def download_resources_json(model_dir, resources_url, resources_branch,
                            resources_version, proxies=None):
    """
    Downloads resources.json to obtain latest packages.
    """
    logger.debug('Downloading resource file...')
    if resources_url == DEFAULT_RESOURCES_URL and resources_branch is not None:
        resources_url = STANZA_RESOURCES_GITHUB + resources_branch
    # handle short name for resources urls; otherwise treat it as url
    if resources_url.lower() in ('stanford', 'stanfordnlp'):
        resources_url = STANFORDNLP_RESOURCES_URL
    # make request
    request_file(
        f'{resources_url}/resources_{resources_version}.json',
        os.path.join(model_dir, 'resources.json'),
        proxies,
        raise_for_status=True
    )


def list_available_languages(model_dir=DEFAULT_MODEL_DIR,
                             resources_url=DEFAULT_RESOURCES_URL,
                             resources_branch=None,
                             resources_version=DEFAULT_RESOURCES_VERSION,
                             proxies=None):
    """
    List the non-alias languages in the resources file
    """
    download_resources_json(model_dir, resources_url, resources_branch,
                            resources_version, proxies)
    with open(os.path.join(model_dir, 'resources.json')) as fin:
        resources = json.load(fin)
    # isinstance(str) is because of fields such as "url"
    # 'alias' is because we want to skip German, alias of de, for example
    languages = [lang for lang in resources
                 if not isinstance(resources[lang], str) and 'alias' not in resources[lang]]
    languages = sorted(languages)
    return languages


# main download function
def download(
        lang='en',
        model_dir=DEFAULT_MODEL_DIR,
        package='default',
        processors={},
        logging_level=None,
        verbose=None,
        resources_url=DEFAULT_RESOURCES_URL,
        resources_branch=None,
        resources_version=DEFAULT_RESOURCES_VERSION,
        model_url=DEFAULT_MODEL_URL,
        proxies=None
    ):
    # set global logging level
    set_logging_level(logging_level, verbose)
    # process different pipeline parameters
    lang, model_dir, package, processors = process_pipeline_parameters(
        lang, model_dir, package, processors
    )

    download_resources_json(model_dir, resources_url, resources_branch,
                            resources_version, proxies)
    # unpack results
    with open(os.path.join(model_dir, 'resources.json')) as fin:
        resources = json.load(fin)
    if lang not in resources:
        raise ValueError(f'Unsupported language: {lang}.')
    if 'alias' in resources[lang]:
        logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
        lang = resources[lang]['alias']
    lang_name = resources[lang]['lang_name'] if 'lang_name' in resources[lang] else ''
    url = resources['url'] if model_url.lower() == 'default' else model_url

    # Default: download zipfile and unzip
    if package == 'default' and (processors is None or len(processors) == 0):
        logger.info(
            f'Downloading default packages for language: {lang} ({lang_name})...'
        )
        request_file(
            f'{url}/{resources_version}/{lang}/default.zip',
            os.path.join(model_dir, lang, f'default.zip'),
            proxies,
            md5=resources[lang]['default_md5'],
        )
        unzip(os.path.join(model_dir, lang), 'default.zip')
    # Customize: maintain download list
    else:
        download_list = maintain_processor_list(
            resources, lang, package, processors
        )
        download_list = add_dependencies(resources, lang, download_list)
        download_list = flatten_processor_list(download_list)
        download_table = make_table(['Processor', 'Package'], download_list)
        logger.info(
            f'Downloading these customized packages for language: '
            f'{lang} ({lang_name})...\n{download_table}'
        )

        # Download packages
        for key, value in download_list:
            try:
                request_file(
                    f'{url}/{resources_version}/{lang}/{key}/{value}.pt',
                    os.path.join(model_dir, lang, key, f'{value}.pt'),
                    proxies,
                    md5=resources[lang][key][value]['md5']
                )
            except KeyError as e:
                raise ValueError(
                    f'Cannot find the following processor and model name combination: '
                    f'{key}, {value}. Please check if you have provided the correct model name.'
                ) from e
    logger.info(f'Finished downloading models and saved to {model_dir}.')
