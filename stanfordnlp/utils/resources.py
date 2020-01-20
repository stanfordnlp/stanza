"""
utilities for getting resources
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path
import json
import hashlib

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
PIPELINE_NAME = ['tokenizer', 'mwt', 'lemmatizer', 'tagger', 'parser', 'nertagger', 'charlm']

# list of language shorthands
conll_shorthands = ['af_afribooms', 'ar_padt', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'cu_proiel', 'da_ddt', 'de_gsd', 'el_gdt', 'en_ewt', 'en_gum', 'en_lines', 'es_ancora', 'et_edt', 'eu_bdt', 'fa_seraji', 'fi_ftb', 'fi_tdt', 'fr_gsd', 'fro_srcmf', 'fr_sequoia', 'fr_spoken', 'ga_idt', 'gl_ctg', 'gl_treegal', 'got_proiel', 'grc_perseus', 'grc_proiel', 'he_htb', 'hi_hdtb', 'hr_set', 'hsb_ufal', 'hu_szeged', 'hy_armtdp', 'id_gsd', 'it_isdt', 'it_postwita', 'ja_gsd', 'kk_ktb', 'kmr_mg', 'ko_gsd', 'ko_kaist', 'la_ittb', 'la_perseus', 'la_proiel', 'lv_lvtb', 'nl_alpino', 'nl_lassysmall', 'no_bokmaal', 'no_nynorsklia', 'no_nynorsk', 'pl_lfg', 'pl_sz', 'pt_bosque', 'ro_rrt', 'ru_syntagrus', 'ru_taiga', 'sk_snk', 'sl_ssj', 'sl_sst', 'sme_giella', 'sr_set', 'sv_lines', 'sv_talbanken', 'tr_imst', 'ug_udt', 'uk_iu', 'ur_udtb', 'vi_vtb', 'zh_gsd']

# all languages with mwt
mwt_languages = ['ar_padt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'de_gsd', 'el_gdt', 'es_ancora', 'fa_seraji', 'fi_ftb', 'fr_gsd', 'fr_sequoia', 'gl_ctg', 'gl_treegal', 'he_htb', 'hy_armtdp', 'it_isdt', 'it_postwita', 'kk_ktb', 'pl_sz', 'pt_bosque', 'tr_imst']

# default treebank for languages
default_treebanks = {'af': 'af_afribooms', 'grc': 'grc_proiel', 'ar': 'ar_padt', 'hy': 'hy_armtdp', 'eu': 'eu_bdt', 'bg': 'bg_btb', 'bxr': 'bxr_bdt', 'ca': 'ca_ancora', 'zh': 'zh_gsd', 'hr': 'hr_set', 'cs': 'cs_pdt', 'da': 'da_ddt', 'nl': 'nl_alpino', 'en': 'en_ewt', 'et': 'et_edt', 'fi': 'fi_tdt', 'fr': 'fr_gsd', 'gl': 'gl_ctg', 'de': 'de_gsd', 'got': 'got_proiel', 'el': 'el_gdt', 'he': 'he_htb', 'hi': 'hi_hdtb', 'hu': 'hu_szeged', 'id': 'id_gsd', 'ga': 'ga_idt', 'it': 'it_isdt', 'ja': 'ja_gsd', 'kk': 'kk_ktb', 'ko': 'ko_kaist', 'kmr': 'kmr_mg', 'la': 'la_ittb', 'lv': 'lv_lvtb', 'sme': 'sme_giella', 'no_bokmaal': 'no_bokmaal', 'no_nynorsk': 'no_nynorsk', 'cu': 'cu_proiel', 'fro': 'fro_srcmf', 'fa': 'fa_seraji', 'pl': 'pl_lfg', 'pt': 'pt_bosque', 'ro': 'ro_rrt', 'ru': 'ru_syntagrus', 'sr': 'sr_set', 'sk': 'sk_snk', 'sl': 'sl_ssj', 'es': 'es_ancora', 'sv': 'sv_talbanken', 'tr': 'tr_imst', 'uk': 'uk_iu', 'hsb': 'hsb_ufal', 'ur': 'ur_udtb', 'ug': 'ug_udt', 'vi': 'vi_vtb'}

# map processor name to file ending
processor_to_ending = {'tokenize': 'tokenizer', 'mwt': 'mwt_expander', 'pos': 'tagger', 'lemma': 'lemmatizer', 'depparse': 'parser', 'ner': 'nertagger'}

# given a language and models path, build a default configuration
def build_default_config(treebank, models_path):
    default_config = {}
    if treebank in mwt_languages:
        default_config['processors'] = 'tokenize,mwt,pos,lemma,depparse,ner'
    else:
        default_config['processors'] = 'tokenize,pos,lemma,depparse,ner'
    if treebank == 'vi_vtb':
        default_config['lemma_use_identity'] = True
        default_config['lemma_batch_size'] = 5000
    treebank_dir = os.path.join(models_path, f"{treebank}_models")
    for processor in default_config['processors'].split(','):
        model_file_ending = f"{processor_to_ending[processor]}.pt"
        default_config[f"{processor}_model_path"] = os.path.join(treebank_dir, f"{treebank}_{model_file_ending}")
        if processor in ['pos', 'depparse']:
            default_config[f"{processor}_pretrain_path"] = os.path.join(treebank_dir, f"{treebank}.pretrain.pt")
        if processor in ['ner']:
            default_config[f"{processor}_charlm_forward_file"] = os.path.join(treebank_dir, f"{treebank}_forward_charlm.pt")
            default_config[f"{processor}_charlm_backward_file"] = os.path.join(treebank_dir, f"{treebank}_backward_charlm.pt")
    return default_config

def ensure_path(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def get_md5(path):
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()

def is_file_existed(path, md5):
    return os.path.exists(path) and get_md5(path) == md5

def download_file(download_url, download_path, verbose=True):
    r = requests.get(download_url, stream=True)
    with open(download_path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        with tqdm(total=file_size, unit='B', unit_scale=True, disable=not verbose) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))

def request_file(download_url, download_path, verbose=True, md5=None):
    ensure_path(download_path)
    if is_file_existed(download_path, md5): 
        if verbose: print(f'File exists: {download_path}.')
        return
    download_file(download_url, download_path, verbose=verbose)
    assert(not md5 or is_file_existed(download_path, md5))

def maintain_download_list(resources, lang, package, processors, verbose):
    download_list = {}
    dependencies = resources[lang][DEFAULT_DEPENDENCIES]
    if processors:
        if verbose: print(f'Processing parameter "processors"...')
        for key, value in processors.items():
            assert(key in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner']) # <TODO>: constant
            assert(isinstance(key, str) and isinstance(value, str))
            if key in resources[lang] and value in resources[lang][key]:
                if verbose: print(f'Find {key}: {value}.')
                download_list[key] = value
            else:
                if verbose: print(f'Can not find {key}: {value}.')

    if package:
        if verbose: print(f'Processing parameter "package"...')
        if package == 'default':
            for key, value in resources[lang][DEFAULT_PROCESSORS].items():
                if key not in download_list:
                    if verbose: print(f'Find {key}: {value}.')
                    download_list[key] = value
        else:
            flag = False
            for key in ['tokenize', 'mwt', 'lemma', 'pos', 'depparse', 'ner']: # <TODO>: constant
                if package in resources[lang][key]:
                    flag = True
                    if key not in download_list:
                        if verbose: print(f'Find {key}: {package}.')
                        download_list[key] = package
                    else:
                        if verbose: print(f'{key}: {package} is overwritten by {key}: {processors[key]}.')
            if not flag and verbose: print((f'Can not find package {package}.'))
    download_list = [(key, value) for key, value in download_list.items()] 
    return download_list


def add_dependencies(resources, lang, download_list, verbose):    
    default_dependencies = resources[lang][DEFAULT_DEPENDENCIES]
    dependencies_list = []
    for key, value in download_list:
        dependencies = default_dependencies.get(key, None)
        dependencies = resources[lang][key][value].get('dependencies', dependencies)
        if dependencies: dependencies_list += [tuple(dependency) for dependency in dependencies]
    dependencies_list = list(set(dependencies_list))
    for key, value in dependencies_list:
        if verbose: print(f'Find dependency {key}: {value}.')
    download_list += dependencies_list
    return download_list

# main download function
def download(lang, dir=None, package='default', processors={}, version=None, verbose=True):
    # If dir and version is None, use default settings.
    if dir is None:
        dir = DEFAULT_MODEL_DIR
    if version is None:
        version = DEFAULT_DOWNLOAD_VERSION
    
    # Download resources.json to obtain latest packages.
    if verbose: print('Downloading resource files...')
    request_file(f'{DEFAULT_RESOURCES_URL}/{DEFAULT_RESOURCES_FILE}', os.path.join(dir, DEFAULT_RESOURCES_FILE), verbose=verbose)
    resources = json.load(open(os.path.join(dir, DEFAULT_RESOURCES_FILE)))
    if lang not in resources:
        print(f'Unsupported language: {lang}.')
        return

    # Default: download zipfile and unzip
    if package == 'default' and len(processors) == 0:
        if verbose: print('Downloading default packages...')
        if verbose: print(f'Downloading {DEFAULT_MODELS_URL}/{version}/{lang}/default.tar.gz...')
        request_file(f'{DEFAULT_MODELS_URL}/{version}/{lang}/default.tar.gz', os.path.join(dir, lang, f'default.tar.gz'), verbose=verbose, md5=resources[lang]['default_md5'])
        # <TODO>: unzip
    # Customize: maintain download list
    else:
        if verbose: print('Downloading customized packages...')
        download_list = maintain_download_list(resources, lang, package, processors, verbose)
        download_list = add_dependencies(resources, lang, download_list, verbose)
        if verbose: print(f'Download list: {download_list}')
        
        # Download packages
        for key, value in download_list:
            if verbose: print(f'Downloading {DEFAULT_MODELS_URL}/{version}/{lang}/{key}/{value}.pt...')
            request_file(f'{DEFAULT_MODELS_URL}/{version}/{lang}/{key}/{value}.pt', os.path.join(dir, lang, key, f'{value}.pt'), verbose=verbose, md5=resources[lang][key][value][MD5])


