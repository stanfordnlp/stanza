"""
utilities for getting resources
"""

import os
import requests
import zipfile

from tqdm import tqdm
from pathlib import Path

# set home dir for default
HOME_DIR = str(Path.home())
DEFAULT_MODEL_DIR = os.path.join(HOME_DIR, 'stanfordnlp_resources')
DEFAULT_MODELS_URL = 'http://nlp.stanford.edu/software/stanfordnlp_models'
DEFAULT_DOWNLOAD_VERSION = 'latest'

# list of language shorthands
conll_shorthands = ['af_afribooms', 'ar_padt', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'cu_proiel', 'da_ddt', 'de_gsd', 'el_gdt', 'en_ewt', 'en_gum', 'en_lines', 'es_ancora', 'et_edt', 'eu_bdt', 'fa_seraji', 'fi_ftb', 'fi_tdt', 'fr_gsd', 'fro_srcmf', 'fr_sequoia', 'fr_spoken', 'ga_idt', 'gl_ctg', 'gl_treegal', 'got_proiel', 'grc_perseus', 'grc_proiel', 'he_htb', 'hi_hdtb', 'hr_set', 'hsb_ufal', 'hu_szeged', 'hy_armtdp', 'id_gsd', 'it_isdt', 'it_postwita', 'ja_gsd', 'kk_ktb', 'kmr_mg', 'ko_gsd', 'ko_kaist', 'la_ittb', 'la_perseus', 'la_proiel', 'lv_lvtb', 'nl_alpino', 'nl_lassysmall', 'no_bokmaal', 'no_nynorsklia', 'no_nynorsk', 'pl_lfg', 'pl_sz', 'pt_bosque', 'ro_rrt', 'ru_syntagrus', 'ru_taiga', 'sk_snk', 'sl_ssj', 'sl_sst', 'sme_giella', 'sr_set', 'sv_lines', 'sv_talbanken', 'tr_imst', 'ug_udt', 'uk_iu', 'ur_udtb', 'vi_vtb', 'zh_gsd']

# all languages with mwt
mwt_languages = ['ar_padt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'de_gsd', 'el_gdt', 'es_ancora', 'fa_seraji', 'fi_ftb', 'fr_gsd', 'fr_sequoia', 'gl_ctg', 'gl_treegal', 'he_htb', 'hy_armtdp', 'it_isdt', 'it_postwita', 'kk_ktb', 'pl_sz', 'pt_bosque', 'tr_imst']

# default treebank for languages
default_treebanks = {'af': 'af_afribooms', 'grc': 'grc_proiel', 'ar': 'ar_padt', 'hy': 'hy_armtdp', 'eu': 'eu_bdt', 'bg': 'bg_btb', 'bxr': 'bxr_bdt', 'ca': 'ca_ancora', 'zh': 'zh_gsd', 'hr': 'hr_set', 'cs': 'cs_pdt', 'da': 'da_ddt', 'nl': 'nl_alpino', 'en': 'en_ewt', 'et': 'et_edt', 'fi': 'fi_tdt', 'fr': 'fr_gsd', 'gl': 'gl_ctg', 'de': 'de_gsd', 'got': 'got_proiel', 'el': 'el_gdt', 'he': 'he_htb', 'hi': 'hi_hdtb', 'hu': 'hu_szeged', 'id': 'id_gsd', 'ga': 'ga_idt', 'it': 'it_isdt', 'ja': 'ja_gsd', 'kk': 'kk_ktb', 'ko': 'ko_kaist', 'kmr': 'kmr_mg', 'la': 'la_ittb', 'lv': 'lv_lvtb', 'sme': 'sme_giella', 'no_bokmaal': 'no_bokmaal', 'no_nynorsk': 'no_nynorsk', 'cu': 'cu_proiel', 'fro': 'fro_srcmf', 'fa': 'fa_seraji', 'pl': 'pl_lfg', 'pt': 'pt_bosque', 'ro': 'ro_rrt', 'ru': 'ru_syntagrus', 'sr': 'sr_set', 'sk': 'sk_snk', 'sl': 'sl_ssj', 'es': 'es_ancora', 'sv': 'sv_talbanken', 'tr': 'tr_imst', 'uk': 'uk_iu', 'hsb': 'hsb_ufal', 'ur': 'ur_udtb', 'ug': 'ug_udt', 'vi': 'vi_vtb'}

# map processor name to file ending
processor_to_ending = {'tokenize': 'tokenizer', 'mwt': 'mwt_expander', 'pos': 'tagger', 'lemma': 'lemmatizer', 'depparse': 'parser'}

# functions for handling configs


# given a language and models path, build a default configuration
def build_default_config(treebank, models_path):
    default_config = {}
    if treebank in mwt_languages:
        default_config['processors'] = 'tokenize,mwt,pos,lemma,depparse'
    else:
        default_config['processors'] = 'tokenize,pos,lemma,depparse'
    if treebank == 'vi_vtb':
        default_config['lemma_use_identity'] = True
        default_config['lemma_batch_size'] = 5000
    treebank_dir = os.path.join(models_path, f"{treebank}_models")
    for processor in default_config['processors'].split(','):
        model_file_ending = f"{processor_to_ending[processor]}.pt"
        default_config[f"{processor}_model_path"] = os.path.join(treebank_dir, f"{treebank}_{model_file_ending}")
        if processor in ['pos', 'depparse']:
            default_config[f"{processor}_pretrain_path"] = os.path.join(treebank_dir, f"{treebank}.pretrain.pt")
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
def download_ud_model(lang_name, resource_dir=None, should_unzip=True, confirm_if_exists=False, force=False,
                      version=DEFAULT_DOWNLOAD_VERSION):
    # ask if user wants to download
    if resource_dir is not None and os.path.exists(os.path.join(resource_dir, f"{lang_name}_models")):
        if confirm_if_exists:
            print("")
            print(f"The model directory already exists at \"{resource_dir}/{lang_name}_models\". Do you want to download the models again? [y/N]")
            should_download = 'y' if force else input()
            should_download = should_download.strip().lower() in ['yes', 'y']
        else:
            should_download = False
    else:
        print('Would you like to download the models for: '+lang_name+' now? (Y/n)')
        should_download = 'y' if force else input()
        should_download = should_download.strip().lower() in ['yes', 'y', '']
    if should_download:
        # set up data directory
        if resource_dir is None:
            print('')
            print('Default download directory: ' + DEFAULT_MODEL_DIR)
            print('Hit enter to continue or type an alternate directory.')
            where_to_download = '' if force else input()
            if where_to_download != '':
                download_dir = where_to_download
            else:
                download_dir = DEFAULT_MODEL_DIR
        else:
            download_dir = resource_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        print('')
        print('Downloading models for: '+lang_name)
        model_zip_file_name = f'{lang_name}_models.zip'
        download_url = f'{DEFAULT_MODELS_URL}/{version}/{model_zip_file_name}'
        download_file_path = os.path.join(download_dir, model_zip_file_name)
        print('Download location: '+download_file_path)

        # initiate download
        r = requests.get(download_url, stream=True)
        with open(download_file_path, 'wb') as f:
            file_size = int(r.headers.get('content-length'))
            default_chunk_size = 67108864
            with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=default_chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
        # unzip models file
        print('')
        print('Download complete.  Models saved to: '+download_file_path)
        if should_unzip:
            unzip_ud_model(lang_name, download_file_path, download_dir)
        # remove the zipe file
        print("Cleaning up...", end="")
        os.remove(download_file_path)
        print('Done.')


# unzip a ud models zip file
def unzip_ud_model(lang_name, zip_file_src, zip_file_target):
    print('Extracting models file for: '+lang_name)
    with zipfile.ZipFile(zip_file_src, "r") as zip_ref:
        zip_ref.extractall(zip_file_target)


# main download function
def download(download_label, resource_dir=None, confirm_if_exists=False, force=False, version=DEFAULT_DOWNLOAD_VERSION):
    if download_label in conll_shorthands:
        download_ud_model(download_label, resource_dir=resource_dir, confirm_if_exists=confirm_if_exists, force=force,
                          version=version)
    elif download_label in default_treebanks:
        print(f'Using the default treebank "{default_treebanks[download_label]}" for language "{download_label}".')
        download_ud_model(default_treebanks[download_label], resource_dir=resource_dir,
                          confirm_if_exists=confirm_if_exists, force=force, version=version)
    else:
        raise ValueError(f'The language or treebank "{download_label}" is not currently supported by this function. Please try again with other languages or treebanks.')
