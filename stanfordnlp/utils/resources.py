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

logger = logging.getLogger(__name__)

# set home dir for default
HOME_DIR = str(Path.home())
DEFAULT_RESOURCES_URL = 'https://raw.githubusercontent.com/stanfordnlp/stanfordnlp/download-refactor/stanfordnlp/utils'
DEFAULT_RESOURCES_FILE = 'resources.json'
DEFAULT_MODEL_DIR = os.path.join(HOME_DIR, 'stanfordnlp_resources')
DEFAULT_MODELS_URL = 'http://nlp.stanford.edu/software/stanza'
DEFAULT_DOWNLOAD_VERSION = 'latest'
DEFAULT_PROCESSORS = "default_processors"
DEFAULT_DEPENDENCIES = "default_dependencies"
PIPELINE_NAMES = ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner']

# given a language and models path, build a default configuration
def build_default_config(resources, lang, dir, load_list):
    default_config = {}
    if lang == 'vi': # <TODO> special handle for vi
        default_config['lemma_use_identity'] = True
        default_config['lemma_batch_size'] = 5000

    for item in load_list:
        processor, model, dependencies = item
        default_config[f"{processor}_model_path"] = os.path.join(dir, lang, processor, model + '.pt')
        if not dependencies: continue
        for dependency in dependencies:
            dep_processor, dep_model = dependency
            if dep_processor == 'charlm': # <TODO>: special handle for charlm
                direction = dep_model.split('_')[1]
                default_config[f"{processor}_{dep_processor}_{direction}_file"] = os.path.join(dir, lang, dep_processor, dep_model + '.pt')
            else:
                default_config[f"{processor}_{dep_processor}_path"] = os.path.join(dir, lang, dep_processor, dep_model + '.pt')

    return default_config

def make_table(header, content, column_width=20):
    '''
    Input:
    header -> List[str]: table header
    content -> List[List[str]]: table content
    column_width -> int: table column width
    
    Output:
    table_str -> str: well-formatted string for the table
    '''
    table_str = ''
    len_column, len_row = len(header), len(content) + 1
    
    table_str += '=' * (len_column * column_width + 1) + '\n'
    
    table_str += '|'
    for item in header: table_str += str(item).ljust(column_width - 1) + '|'
    table_str += '\n'
    
    table_str += '-' * (len_column * column_width + 1) + '\n'
    
    for line in content:
        table_str += '|'
        for item in line:
            table_str += str(item).ljust(column_width - 1) + '|'
        table_str += '\n'
    
    table_str += '=' * (len_column * column_width + 1) + '\n'
    
    return table_str

def ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)

def get_md5(path):
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()

def unzip(dir, filename):
    logger.debug(f'Unzip: {dir}/{filename}...')
    with zipfile.ZipFile(os.path.join(dir, filename)) as f:
        f.extractall(dir)

def is_file_existed(path, md5):
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
    if is_file_existed(path, md5): 
        logger.info(f'File exists: {path}.')
        return
    download_file(url, path)
    assert(not md5 or is_file_existed(path, md5))

def maintain_download_list(resources, lang, package, processors):
    download_list = {}
    dependencies = resources[lang][DEFAULT_DEPENDENCIES]
    if processors:
        logger.debug(f'Processing parameter "processors"...')
        for key, value in processors.items():
            assert(key in PIPELINE_NAMES)
            assert(isinstance(key, str) and isinstance(value, str))
            if key in resources[lang] and value in resources[lang][key]:
                logger.debug(f'Find {key}: {value}.')
                download_list[key] = value
            else:
                logger.warning(f'Can not find {key}: {value}.')

    if package:
        logger.debug(f'Processing parameter "package"...')
        if package == 'default':
            for key, value in resources[lang][DEFAULT_PROCESSORS].items():
                if key not in download_list:
                    logger.debug(f'Find {key}: {value}.')
                    download_list[key] = value
        else:
            flag = False
            for key in PIPELINE_NAMES:
                if package in resources[lang][key]:
                    flag = True
                    if key not in download_list:
                        logger.debug(f'Find {key}: {package}.')
                        download_list[key] = package
                    else:
                        logger.debug(f'{key}: {package} is overwritten by {key}: {processors[key]}.')
            if not flag: logger.warning((f'Can not find package: {package}.'))
    download_list = [[key, value] for key, value in download_list.items()] 
    return download_list

def add_dependencies(resources, lang, download_list):    
    default_dependencies = resources[lang][DEFAULT_DEPENDENCIES]
    dependencies_list = []
    for key, value in download_list:
        dependencies = default_dependencies.get(key, None)
        dependencies = resources[lang][key][value].get('dependencies', dependencies)
        if dependencies: dependencies_list += [tuple(dependency) for dependency in dependencies]
    dependencies_list = [list(i) for i in set(dependencies_list)]
    for key, value in dependencies_list:
        logger.debug(f'Find dependency {key}: {value}.')
    download_list += dependencies_list
    return download_list

# main download function
def download(lang, dir=DEFAULT_MODEL_DIR, package='default', processors={}, version=DEFAULT_DOWNLOAD_VERSION, logging_level='INFO'):
    assert logging_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    logger.setLevel(logging_level)
    
    # Download resources.json to obtain latest packages.
    logger.info('Downloading resource file...')
    request_file(f'{DEFAULT_RESOURCES_URL}/{DEFAULT_RESOURCES_FILE}', os.path.join(dir, DEFAULT_RESOURCES_FILE))
    resources = json.load(open(os.path.join(dir, DEFAULT_RESOURCES_FILE)))
    if lang not in resources:
        logger.warning(f'Unsupported language: {lang}.')
        return

    # Default: download zipfile and unzip
    if package == 'default' and len(processors) == 0:
        logger.info('Downloading default packages...')
        request_file(f'{DEFAULT_MODELS_URL}/{version}/{lang}/default.zip', os.path.join(dir, lang, f'default.zip'), md5=resources[lang]['default_md5'])
        unzip(os.path.join(dir, lang), 'default.zip')
    # Customize: maintain download list
    else:
        logger.info('Downloading customized packages...')
        download_list = maintain_download_list(resources, lang, package, processors)
        download_list = add_dependencies(resources, lang, download_list)
        download_table = make_table(['Processor', 'Model'], download_list)
        logger.info(f'Download list:\n{download_table}')
        
        # Download packages
        for key, value in download_list:
            request_file(f'{DEFAULT_MODELS_URL}/{version}/{lang}/{key}/{value}.pt', os.path.join(dir, lang, key, f'{value}.pt'), md5=resources[lang][key][value]['md5'])
