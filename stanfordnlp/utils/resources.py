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
MD5 = 'md5'
# PIPELINE_NAME = ['tokenizer', 'mwt', 'lemmatizer', 'tagger', 'parser', 'nertagger', 'charlm']

# # list of language shorthands
# conll_shorthands = ['af_afribooms', 'ar_padt', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'cu_proiel', 'da_ddt', 'de_gsd', 'el_gdt', 'en_ewt', 'en_gum', 'en_lines', 'es_ancora', 'et_edt', 'eu_bdt', 'fa_seraji', 'fi_ftb', 'fi_tdt', 'fr_gsd', 'fro_srcmf', 'fr_sequoia', 'fr_spoken', 'ga_idt', 'gl_ctg', 'gl_treegal', 'got_proiel', 'grc_perseus', 'grc_proiel', 'he_htb', 'hi_hdtb', 'hr_set', 'hsb_ufal', 'hu_szeged', 'hy_armtdp', 'id_gsd', 'it_isdt', 'it_postwita', 'ja_gsd', 'kk_ktb', 'kmr_mg', 'ko_gsd', 'ko_kaist', 'la_ittb', 'la_perseus', 'la_proiel', 'lv_lvtb', 'nl_alpino', 'nl_lassysmall', 'no_bokmaal', 'no_nynorsklia', 'no_nynorsk', 'pl_lfg', 'pl_sz', 'pt_bosque', 'ro_rrt', 'ru_syntagrus', 'ru_taiga', 'sk_snk', 'sl_ssj', 'sl_sst', 'sme_giella', 'sr_set', 'sv_lines', 'sv_talbanken', 'tr_imst', 'ug_udt', 'uk_iu', 'ur_udtb', 'vi_vtb', 'zh_gsd']

# # all languages with mwt
# mwt_languages = ['ar_padt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'de_gsd', 'el_gdt', 'es_ancora', 'fa_seraji', 'fi_ftb', 'fr_gsd', 'fr_sequoia', 'gl_ctg', 'gl_treegal', 'he_htb', 'hy_armtdp', 'it_isdt', 'it_postwita', 'kk_ktb', 'pl_sz', 'pt_bosque', 'tr_imst']

# # default treebank for languages
# default_treebanks = {'af': 'af_afribooms', 'grc': 'grc_proiel', 'ar': 'ar_padt', 'hy': 'hy_armtdp', 'eu': 'eu_bdt', 'bg': 'bg_btb', 'bxr': 'bxr_bdt', 'ca': 'ca_ancora', 'zh': 'zh_gsd', 'hr': 'hr_set', 'cs': 'cs_pdt', 'da': 'da_ddt', 'nl': 'nl_alpino', 'en': 'en_ewt', 'et': 'et_edt', 'fi': 'fi_tdt', 'fr': 'fr_gsd', 'gl': 'gl_ctg', 'de': 'de_gsd', 'got': 'got_proiel', 'el': 'el_gdt', 'he': 'he_htb', 'hi': 'hi_hdtb', 'hu': 'hu_szeged', 'id': 'id_gsd', 'ga': 'ga_idt', 'it': 'it_isdt', 'ja': 'ja_gsd', 'kk': 'kk_ktb', 'ko': 'ko_kaist', 'kmr': 'kmr_mg', 'la': 'la_ittb', 'lv': 'lv_lvtb', 'sme': 'sme_giella', 'no_bokmaal': 'no_bokmaal', 'no_nynorsk': 'no_nynorsk', 'cu': 'cu_proiel', 'fro': 'fro_srcmf', 'fa': 'fa_seraji', 'pl': 'pl_lfg', 'pt': 'pt_bosque', 'ro': 'ro_rrt', 'ru': 'ru_syntagrus', 'sr': 'sr_set', 'sk': 'sk_snk', 'sl': 'sl_ssj', 'es': 'es_ancora', 'sv': 'sv_talbanken', 'tr': 'tr_imst', 'uk': 'uk_iu', 'hsb': 'hsb_ufal', 'ur': 'ur_udtb', 'ug': 'ug_udt', 'vi': 'vi_vtb'}

# map processor name to file ending
processor_to_ending = {'tokenize': 'tokenizer', 'mwt': 'mwt_expander', 'pos': 'tagger', 'lemma': 'lemmatizer', 'depparse': 'parser', 'ner': 'nertagger'}

# # given a language and models path, build a default configuration
# def build_default_config(treebank, models_path):
#     default_config = {}
#     if treebank in mwt_languages:
#         default_config['processors'] = 'tokenize,mwt,pos,lemma,depparse,ner'
#     else:
#         default_config['processors'] = 'tokenize,pos,lemma,depparse,ner'
#     if treebank == 'vi_vtb':
#         default_config['lemma_use_identity'] = True
#         default_config['lemma_batch_size'] = 5000
#     lang, model = treebank.split('_')
#     print(f'lang: {lang}, model: {model}')
#     for processor in default_config['processors'].split(','):
#         model_file_ending = f"{processor_to_ending[processor]}.pt"
#         default_config[f"{processor}_model_path"] = os.path.join(models_path, lang, processor, model + '.pt')
#         if processor in ['pos', 'depparse']:
#             default_config[f"{processor}_pretrain_path"] = os.path.join(models_path, lang, 'pretrain', model + '.pt')
#         if processor in ['ner']:
#             default_config[f"{processor}_charlm_forward_file"] = os.path.join(models_path, lang, 'charlm', model + '_forward.pt')
#             default_config[f"{processor}_charlm_backward_file"] = os.path.join(models_path, lang, 'charlm', model + '_backward.pt')
#     print(default_config)
#     return default_config

# given a language and models path, build a default configuration
def build_default_config(resources, lang, dir, load_list):
    default_config = {}
    if lang == 'vi': # <TODO> identical ? treebank == 'vi_vtb':
        default_config['lemma_use_identity'] = True
        default_config['lemma_batch_size'] = 5000

    for item in load_list:
        processor, model, dependencies = item
        default_config[f"{processor}_model_path"] = os.path.join(dir, lang, processor, model + '.pt')
        if not dependencies: continue
        for dependency in dependencies:
            dep_processor, dep_model = dependency
            if dep_processor == 'charlm': # <TODO>: special handle
                direction = dep_model.split('_')[1]
                default_config[f"{processor}_{dep_processor}_{direction}_file"] = os.path.join(dir, lang, dep_processor, dep_model + '.pt')
            else:
                default_config[f"{processor}_{dep_processor}_path"] = os.path.join(dir, lang, dep_processor, dep_model + '.pt')

    # print(default_config)
    return default_config

def ensure_path(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def get_md5(path):
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()

def is_file_existed(path, md5):
    return os.path.exists(path) and get_md5(path) == md5

def download_file(download_url, download_path):
    verbose = logger.level in [0, 10, 20]
    r = requests.get(download_url, stream=True)
    with open(download_path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + download_url
        with tqdm(total=file_size, unit='B', unit_scale=True, disable=not verbose, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))

def unzip(path, filename):
    logger.debug(f'Unzip: {path}/{filename}...')
    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        f.extractall(path)


def request_file(download_url, download_path, md5=None):
    ensure_path(download_path)
    if is_file_existed(download_path, md5): 
        logger.info(f'File exists: {download_path}.')
        return
    download_file(download_url, download_path)
    assert(not md5 or is_file_existed(download_path, md5))

def maintain_download_list(resources, lang, package, processors):
    download_list = {}
    dependencies = resources[lang][DEFAULT_DEPENDENCIES]
    if processors:
        logger.debug(f'Processing parameter "processors"...')
        for key, value in processors.items():
            assert(key in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner']) # <TODO>: constant
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
            for key in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner']: # <TODO>: constant
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


# main download function
def download(lang, dir=None, package='default', processors={}, version=None, logging_level='INFO'):
    assert logging_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    logger.setLevel(logging_level)
    # If dir and version is None, use default settings.
    if dir is None:
        dir = DEFAULT_MODEL_DIR
    if version is None:
        version = DEFAULT_DOWNLOAD_VERSION
    
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
            request_file(f'{DEFAULT_MODELS_URL}/{version}/{lang}/{key}/{value}.pt', os.path.join(dir, lang, key, f'{value}.pt'), md5=resources[lang][key][value][MD5])


