"""
Common utilities for Stanza resources.
"""

from collections import defaultdict, namedtuple
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import requests
import shutil
import tempfile
import zipfile

from tqdm.auto import tqdm

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

PRETRAIN_NAMES = ("pretrain", "forward_charlm", "backward_charlm")

class ResourcesFileNotFoundError(FileNotFoundError):
    def __init__(self, resources_filepath):
        super().__init__(f"Resources file not found at: {resources_filepath}  Try to download the model again.")
        self.resources_filepath = resources_filepath

class UnknownLanguageError(ValueError):
    def __init__(self, unknown):
        super().__init__(f"Unknown language requested: {unknown}")
        self.unknown_language = unknown

class UnknownProcessorError(ValueError):
    def __init__(self, unknown):
        super().__init__(f"Unknown processor type requested: {unknown}")
        self.unknown_processor = unknown

ModelSpecification = namedtuple('ModelSpecification', ['processor', 'package', 'dependencies'])

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

def assert_file_exists(path, md5=None, alternate_md5=None):
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, "Cannot find expected file", path)
    if md5:
        file_md5 = get_md5(path)
        if file_md5 != md5:
            if file_md5 == alternate_md5:
                logger.debug("Found a possibly older version of file %s, md5 %s instead of %s", path, alternate_md5, md5)
            else:
                raise ValueError("md5 for %s is %s, expected %s" % (path, file_md5, md5))

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

def request_file(url, path, proxies=None, md5=None, raise_for_status=False, log_info=True, alternate_md5=None):
    """
    A complete wrapper over download_file() that also make sure the directory of
    `path` exists, and that a file matching the md5 value does not exist.

    alternate_md5 allows for an alternate md5 that is acceptable (such as if an older version of a file is okay)
    """
    basedir = Path(path).parent
    ensure_dir(basedir)
    if file_exists(path, md5):
        if log_info:
            logger.info(f'File exists: {path}')
        else:
            logger.debug(f'File exists: {path}')
        return
    # We write data first to a temporary directory,
    # then use os.replace() so that multiple processes
    # running at the same time don't clobber each other
    # with partially downloaded files
    # This was especially common with resources.json
    with tempfile.TemporaryDirectory(dir=basedir) as temp:
        temppath = os.path.join(temp, os.path.split(path)[-1])
        download_file(url, temppath, proxies, raise_for_status)
        os.replace(temppath, path)
    assert_file_exists(path, md5, alternate_md5)

def sort_processors(processor_list):
    sorted_list = []
    for processor in PIPELINE_NAMES:
        for item in processor_list:
            if item[0] == processor:
                sorted_list.append(item)
    # going just by processors in PIPELINE_NAMES, this drops any names
    # which are not an official processor but might still be useful
    # check the list and append them to the end
    # this is especially useful when downloading pretrain or charlm models
    for processor in processor_list:
        for item in sorted_list:
            if processor[0] == item[0]:
                break
        else:
            sorted_list.append(item)
    return sorted_list

def add_mwt(processors, resources, lang):
    """Add mwt if tokenize is passed without mwt.

    If tokenize is in the list, but mwt is not, and there is a corresponding
    tokenize and mwt pair in the resources file, mwt is added so no missing
    mwt errors are raised.
    """
    value = processors[TOKENIZE]
    if value == "default" and MWT in resources[lang]['default_processors']:
        logger.warning("Language %s package default expects mwt, which has been added", lang)
        processors[MWT] = 'default'
    elif (value in resources[lang][TOKENIZE]
          and MWT in resources[lang]
          and value in resources[lang][MWT]):
        logger.warning("Language %s package %s expects mwt, which has been added", lang, value)
        processors[MWT] = value

def maintain_processor_list(resources, lang, package, processors, allow_pretrain=False):
    """
    Given a parsed resources file, language, and possible package
    and/or processors, expands the package to the list of processors

    Returns a list of processors
    Each item in the list of processors is a pair:
      name, then a list of ModelSpecification
    so, for example:
      [['pos', [ModelSpecification(processor='pos', package='gsd', dependencies=None)]],
       ['depparse', [ModelSpecification(processor='depparse', package='gsd', dependencies=None)]]]
    """
    processor_list = defaultdict(list)
    # resolve processor models
    if processors:
        logger.debug(f'Processing parameter "processors"...')
        if TOKENIZE in processors and MWT not in processors:
            add_mwt(processors, resources, lang)
        for key, plist in processors.items():
            if not isinstance(key, str):
                raise ValueError("Processor names must be strings")
            if not isinstance(plist, (tuple, list, str)):
                raise ValueError("Processor values must be strings")
            if isinstance(plist, str):
                plist = [plist]
            if key not in PIPELINE_NAMES:
                if not allow_pretrain or key not in PRETRAIN_NAMES:
                    raise UnknownProcessorError(key)
            for value in plist:
                # check if keys and values can be found
                if key in resources[lang] and value in resources[lang][key]:
                    logger.debug(f'Found {key}: {value}.')
                    processor_list[key].append(value)
                # allow values to be default in some cases
                elif key in resources[lang]['default_processors'] and value == 'default':
                    logger.debug(
                        f'Found {key}: {resources[lang]["default_processors"][key]}.'
                    )
                    processor_list[key].append(resources[lang]['default_processors'][key])
                # allow processors to be set to variants that we didn't implement
                elif value in PROCESSOR_VARIANTS[key]:
                    logger.debug(
                        f'Found {key}: {value}. '
                        f'Using external {value} variant for the {key} processor.'
                    )
                    processor_list[key].append(value)
                # allow lemma to be set to "identity"
                elif key == LEMMA and value == 'identity':
                    logger.debug(
                        f'Found {key}: {value}. Using identity lemmatizer.'
                    )
                    processor_list[key].append(value)
                # not a processor in the officially supported processor list
                elif key not in resources[lang]:
                    logger.debug(
                        f'{key}: {value} is not officially supported by Stanza, '
                        f'loading it anyway.'
                    )
                    processor_list[key].append(value)
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
                    processor_list[key].append(value)
        else:
            flag = False
            for key in PIPELINE_NAMES:
                if key not in resources[lang]: continue
                if package in resources[lang][key]:
                    flag = True
                    if key not in processor_list:
                        logger.debug(f'Found {key}: {package}.')
                        processor_list[key].append(package)
                    else:
                        logger.debug(
                            f'{key}: {package} is overwritten by '
                            f'{key}: {processors[key]}.'
                        )
            if not flag: logger.warning((f'Can not find package: {package}.'))
    processor_list = [[key, [ModelSpecification(processor=key, package=value, dependencies=None) for value in plist]] for key, plist in processor_list.items()]
    processor_list = sort_processors(processor_list)
    return processor_list

def add_dependencies(resources, lang, processor_list):
    """
    Expand the processor_list as given in maintain_processor_list to have the dependencies

    Still a list of model types to ModelSpecifications
    the dependencies are tuples: name and package
    for example:
    [['pos', (ModelSpecification(processor='pos', package='gsd', dependencies=(('pretrain', 'gsd'),)),)],
     ['depparse', (ModelSpecification(processor='depparse', package='gsd', dependencies=(('pretrain', 'gsd'),)),)]]
    """
    default_dependencies = resources[lang]['default_dependencies']
    for item in processor_list:
        processor, model_specs = item
        new_model_specs = []
        for model_spec in model_specs:
            dependencies = default_dependencies.get(processor, None)
            # skip dependency checking for external variants of processors and identity lemmatizer
            if not any([
                    model_spec.package in PROCESSOR_VARIANTS[processor],
                    processor == LEMMA and model_spec.package == 'identity'
                ]):
                dependencies = resources[lang].get(processor, {}).get(model_spec.package, {}).get('dependencies', dependencies)
            if dependencies:
                dependencies = [(dependency['model'], dependency['package']) for dependency in dependencies]
                model_spec = model_spec._replace(dependencies=tuple(dependencies))
            new_model_specs.append(model_spec)
        item[1] = tuple(new_model_specs)
    return processor_list

def flatten_processor_list(processor_list):
    """
    The flattened processor list is just a list of types & packages

    For example:
      [['pos', 'gsd'], ['depparse', 'gsd'], ['pretrain', 'gsd']]
    """
    flattened_processor_list = []
    dependencies_list = []
    for item in processor_list:
        processor, model_specs = item
        for model_spec in model_specs:
            package = model_spec.package
            dependencies = model_spec.dependencies
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

    if isinstance(processors, (str, list, tuple)):
        # Special case: processors is str, compatible with older version
        # also allow for setting alternate packages for these processors
        # via the package argument
        if package is None:
            # each processor will be 'default' for this language
            package = defaultdict(lambda: 'default')
        elif isinstance(package, str):
            # same, but now the named package will be the default instead
            default = package
            package = defaultdict(lambda: default)
        elif isinstance(package, dict):
            # the dictionary of packages will be used to build the processors dict
            # any processor not specified in package will be 'default'
            package = defaultdict(lambda: 'default', package)
        else:
            raise TypeError(
                f"The parameter 'package' should be None, str, or dict, "
                f"but got {type(package).__name__} instead."
            )
        if isinstance(processors, str):
            processors = [x.strip().lower() for x in processors.split(",")]
        processors = {
            processor: package[processor] for processor in processors
        }
        package = None
    elif isinstance(processors, dict):
        processors = {
            k.strip().lower(): ([v_i.strip().lower() for v_i in v] if isinstance(v, (tuple, list)) else v.strip().lower())
            for k, v in processors.items()
        }
    elif processors is not None:
        raise TypeError(
            f"The parameter 'processors' should be dict or str, "
            f"but got {type(processors).__name__} instead."
        )

    if isinstance(package, str):
        package = package.strip().lower()
    elif package is not None:
        raise TypeError(
            f"The parameter 'package' should be str, or a dict if 'processors' is a str, "
            f"but got {type(package).__name__} instead."
        )

    return lang, model_dir, package, processors

def download_resources_json(model_dir=DEFAULT_MODEL_DIR,
                            resources_url=DEFAULT_RESOURCES_URL,
                            resources_branch=None,
                            resources_version=DEFAULT_RESOURCES_VERSION,
                            proxies=None):
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


def load_resources_json(model_dir=DEFAULT_MODEL_DIR):
    """
    Unpack the resources json file from the given model_dir
    """
    resources_filepath = os.path.join(model_dir, 'resources.json')
    if not os.path.exists(resources_filepath):
        raise ResourcesFileNotFoundError(resources_filepath)
    with open(resources_filepath) as fin:
        resources = json.load(fin)
    return resources


def list_available_languages(model_dir=DEFAULT_MODEL_DIR,
                             resources_url=DEFAULT_RESOURCES_URL,
                             resources_branch=None,
                             resources_version=DEFAULT_RESOURCES_VERSION,
                             proxies=None):
    """
    List the non-alias languages in the resources file
    """
    download_resources_json(model_dir, resources_url, resources_branch, resources_version, proxies)
    resources = load_resources_json(model_dir)
    # isinstance(str) is because of fields such as "url"
    # 'alias' is because we want to skip German, alias of de, for example
    languages = [lang for lang in resources
                 if not isinstance(resources[lang], str) and 'alias' not in resources[lang]]
    languages = sorted(languages)
    return languages

def expand_model_url(resources, model_url):
    """
    Returns the url in the resources dict if model_url is default, or returns the model_url
    """
    return resources['url'] if model_url.lower() == 'default' else model_url

def download_models(download_list,
                    resources,
                    lang,
                    model_dir=DEFAULT_MODEL_DIR,
                    resources_version=DEFAULT_RESOURCES_VERSION,
                    model_url=DEFAULT_MODEL_URL,
                    proxies=None,
                    log_info=True):
    lang_name = resources.get(lang, {}).get('lang_name', lang)
    download_table = make_table(['Processor', 'Package'], download_list)
    if log_info:
        log_msg = logger.info
    else:
        log_msg = logger.debug
    log_msg(
        f'Downloading these customized packages for language: '
        f'{lang} ({lang_name})...\n{download_table}'
    )

    url = expand_model_url(resources, model_url)

    # Download packages
    for key, value in download_list:
        try:
            request_file(
                url.format(resources_version=resources_version, lang=lang, filename=f"{key}/{value}.pt"),
                os.path.join(model_dir, lang, key, f'{value}.pt'),
                proxies,
                md5=resources[lang][key][value]['md5'],
                log_info=log_info,
                alternate_md5=resources[lang][key][value].get('alternate_md5', None)
            )
        except KeyError as e:
            raise ValueError(
                f'Cannot find the following processor and model name combination: '
                f'{key}, {value}. Please check if you have provided the correct model name.'
            ) from e

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

    download_resources_json(model_dir, resources_url, resources_branch, resources_version, proxies)
    resources = load_resources_json(model_dir)
    if lang not in resources:
        raise UnknownLanguageError(lang)
    if 'alias' in resources[lang]:
        logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
        lang = resources[lang]['alias']
    lang_name = resources.get(lang, {}).get('lang_name', lang)
    url = expand_model_url(resources, model_url)

    # Default: download zipfile and unzip
    if package == 'default' and (processors is None or len(processors) == 0):
        logger.info(
            f'Downloading default packages for language: {lang} ({lang_name}) ...'
        )
        # want the URL to become, for example:
        # https://huggingface.co/stanfordnlp/stanza-af/resolve/v1.3.0/models/default.zip
        # so we hopefully start from
        # https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/v{resources_version}/models/{filename}
        request_file(
            url.format(resources_version=resources_version, lang=lang, filename="default.zip"),
            os.path.join(model_dir, lang, f'default.zip'),
            proxies,
            md5=resources[lang]['default_md5'],
        )
        unzip(os.path.join(model_dir, lang), 'default.zip')
    # Customize: maintain download list
    else:
        download_list = maintain_processor_list(resources, lang, package, processors, allow_pretrain=True)
        download_list = add_dependencies(resources, lang, download_list)
        download_list = flatten_processor_list(download_list)
        download_models(download_list=download_list,
                        resources=resources,
                        lang=lang,
                        model_dir=model_dir,
                        resources_version=resources_version,
                        model_url=model_url,
                        proxies=proxies,
                        log_info=True)
    logger.info(f'Finished downloading models and saved to {model_dir}.')
