"""
utilities for getting resources
"""

import os
import requests
import subprocess
import sys
import urllib.request
import zipfile

from clint.textui import progress
from pathlib import Path

# set home dir for default
HOME_DIR = str(Path.home())

# list of language shorthands
conll_shorthands = ['af_afribooms', 'ar_padt', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'cu_proiel', 'da_ddt', 'de_gsd', 'el_gdt', 'en_ewt', 'en_gum', 'en_lines', 'es_ancora', 'et_edt', 'eu_bdt', 'fa_seraji', 'fi_ftb', 'fi_tdt', 'fr_gsd', 'fro_srcmf', 'fr_sequoia', 'fr_spoken', 'ga_idt', 'gl_ctg', 'gl_treegal', 'got_proiel', 'grc_perseus', 'grc_proiel', 'he_htb', 'hi_hdtb', 'hr_set', 'hsb_ufal', 'hu_szeged', 'hy_armtdp', 'id_gsd', 'it_isdt', 'it_postwita', 'ja_gsd', 'kk_ktb', 'kmr_mg', 'ko_gsd', 'ko_kaist', 'la_ittb', 'la_perseus', 'la_proiel', 'lv_lvtb', 'nl_alpino', 'nl_lassysmall', 'no_bokmaal', 'no_nynorsklia', 'no_nynorsk', 'pl_lfg', 'pl_sz', 'pt_bosque', 'ro_rrt', 'ru_syntagrus', 'ru_taiga', 'sk_snk', 'sl_ssj', 'sl_sst', 'sme_giella', 'sr_set', 'sv_lines', 'sv_talbanken', 'tr_imst', 'ug_udt', 'uk_iu', 'ur_udtb', 'vi_vtb', 'zh_gsd']

# all languages with mwt
mwt_languages = ['ar_padt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'de_gsd', 'el_gdt', 'es_ancora', 'fa_seraji', 'fi_ftb', 'fr_gsd', 'fr_sequoia', 'gl_ctg', 'gl_treegal', 'he_htb', 'hy_armtdp', 'it_isdt', 'it_postwita', 'kk_ktb', 'pl_sz', 'pt_bosque', 'tr_imst']

# map processor name to file ending
processor_to_ending = {'tokenize': 'tokenizer', 'mwt': 'mwt_expander', 'pos': 'tagger', 'lemma': 'lemmatizer', 'depparse': 'parser'}

# functions for handling configs

# given a language and models path, build a default configuration
def build_default_config(lang, models_path):
    default_config = {}
    if lang in mwt_languages:
        default_config['processors'] = 'tokenize,mwt,pos,lemma,depparse'
    else:
        default_config['processors'] = 'tokenize,pos,lemma,depparse'
    lang_dir = models_path+'/'+lang+'_models'
    for processor in default_config['processors'].split(','):
        model_file_ending = processor_to_ending[processor]+'.pt'
        default_config[processor+'.model_path'] = lang_dir+'/'+lang+'_'+model_file_ending
        if processor in ['pos', 'depparse']:
            pretrain_file_ending = processor_to_ending[processor]+'.pretrain.pt'
            default_config[processor+'.pretrain_path'] = lang_dir+'/'+lang+'_'+pretrain_file_ending
    return default_config


# load a config from file
def load_config(config_file_path):
    loaded_config = {}
    with open(config_file_path) as config_file:
        for config_line in config_file:
            config_key, config_value = config_line.split(':')
            loaded_config[config_key] = config_value.rstrip().lstrip()
    return loaded_config


# download a ud models zip file
def download_ud_model(lang_name, resource_dir=HOME_DIR+'/stanfordnlp_resources', should_unzip=True):
    # ask if user wants to download
    print('')
    print('Would you like to download the models for: '+lang_name+' now? (yes/no)')
    should_download = input()
    if should_download in ['yes', 'y']:
        # set up data directory
        download_dir = resource_dir
        print('')
        print('Default download directory: '+download_dir)
        print('Hit enter to continue or type an alternate directory.')
        where_to_download = input()
        if where_to_download != '':
            download_dir = where_to_download
        # initiate download
        if not os.path.exists(download_dir):
            subprocess.call('mkdir '+download_dir, shell=True)
        print('')
        print('Downloading models for: '+lang_name)
        model_zip_file_name = lang_name+'_models.zip'
        download_url = 'http://nlp.stanford.edu/software/conll_2018/'+model_zip_file_name
        download_file_path = download_dir+'/'+model_zip_file_name
        print('Download location: '+download_file_path)
        r = requests.get(download_url, stream=True)
        with open(download_file_path, 'wb') as f:
            file_size = int(r.headers.get('content-length'))
            default_chunk_size = 67108864
            for chunk in progress.bar(r.iter_content(chunk_size=default_chunk_size), expected_size=(file_size/default_chunk_size)+1):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # unzip models file
        print('')
        print('Download complete.  Models saved to: '+download_file_path)
        if should_unzip:
            unzip_ud_model(lang_name, download_file_path, download_dir)
        # remove the zipe file
        subprocess.call('rm '+download_file_path, shell=True)
        print('Deleted zipfile.')


# unzip a ud models zip file
def unzip_ud_model(lang_name, zip_file_src, zip_file_target):
    print('Unzipping models file for: '+lang_name)
    with zipfile.ZipFile(zip_file_src, "r") as zip_ref:
        zip_ref.extractall(zip_file_target)


# main download function
def download(download_label, resource_dir=HOME_DIR+'/stanfordnlp_resources'):
    if download_label in conll_shorthands:
        download_ud_model(download_label, resource_dir=resource_dir)
