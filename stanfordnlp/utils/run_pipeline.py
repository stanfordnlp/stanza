"""
script for running full pipeline on command line
"""

import argparse
import os
import subprocess
import sys
import urllib.request

from pathlib import Path
from stanfordnlp.pipeline import Document, Pipeline

# all languages with mwt
MWT_LANGUAGES = ['ar_padt', 'ca_ancora', 'cs_cac', 'cs_fictree', 'cs_pdt', 'de_gsd', 'el_gdt', 'es_ancora', 'fa_seraji', 'fi_ftb', 'fr_gsd', 'fr_sequoia', 'gl_ctg', 'gl_treegal', 'he_htb', 'hy_armtdp', 'it_isdt', 'it_postwita', 'kk_ktb', 'pl_sz', 'pt_bosque', 'tr_imst']

# map processor name to file ending
processor_to_ending = {'tokenize': 'tokenizer', 'lemma': 'lemmatizer', 'pos': 'tagger', 'depparse': 'parser'}

# given a language and models path, build a default configuration
def build_default_config(lang,models_path):
    default_config = {}
    if lang in MWT_LANGUAGES:
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


# ensure a model is present
def ensure_models_for_lang(lang, model_data_path):
    # create model data dir if necessary
    if not os.path.exists(model_data_path):
        subprocess.call('mkdir '+model_data_path, shell=True)
    # check for the tokenizer
    if not os.path.exists(model_data_path+'/'+lang+'_models/'+lang+'_tokenizer.pt'):
        print('Could not find models for: '+lang)
        print('Would you like to download the models for: '+lang+' now? (yes/no)')
        response = input()
        if response in ['yes', 'y']:
            # download the models zip file
            model_zip_file_name = lang+'_models.zip'
            download_url = 'http://nlp.stanford.edu/software/conll_2018/'+model_zip_file_name
            urllib.request.urlretrieve(download_url, model_data_path+'/'+model_zip_file_name)
            # unzip the models
            subprocess.call('cd '+model_data_path+' ; unzip '+model_zip_file_name, shell=True)
            # delete the zip file
            subprocess.call('rm '+model_data_path+'/'+model_zip_file_name, shell=True)
        else:
            print('Okay the models will not be downloaded at this time.  Please download the models before running the pipeline.  Exiting...')
            sys.exit()


if __name__ == '__main__':
    # get arguments
    # determine home directory
    home_dir = str(Path.home())
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='pipeline config file | default: None', default=None)
    parser.add_argument('-m', '--models_dir', help='location of models files | default: ~/stanfordnlp_data', 
                        default=home_dir+'/stanfordnlp_data')
    parser.add_argument('-l', '--language', help='language of text | default: en_ewt', default='en_ewt')
    parser.add_argument('text_file')
    args = parser.parse_args()
    # set up output file
    output_file_path = args.text_file+'.out'
    # check for models
    ensure_models_for_lang(args.language,args.models_dir)
    # set up pipeline
    if args.config is not None:
        print('loading pipeline configs from: '+args.config)
        pipeline_config = load_config(args.config)
    else:
        print('using default pipeline configs for: '+args.language)
        pipeline_config = build_default_config(args.language, args.models_dir)
    print('using following config for pipeline: ')
    print(pipeline_config)
    pipeline = Pipeline(config=pipeline_config)
    # run process
    # load input text
    input_text = open(args.text_file).read()
    # build document
    doc = Document(input_text)
    # run pipeline 
    print('running pipeline...')
    pipeline.process(doc)
    # write conll to file
    doc.write_conll_to_file(output_file_path)
    print('done.  results written to: '+output_file_path)
