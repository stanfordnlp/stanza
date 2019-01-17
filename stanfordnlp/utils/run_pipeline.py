import argparse
import os

from stanfordnlp.pipeline import Document, Pipeline

english_config = {
    'processors': 'tokenize,pos,lemma,depparse',
    'tokenize.model_path': 'saved_models/tokenize/en_ewt_tokenizer.pt',
    'lemma.model_path': 'saved_models/lemma/en_ewt_lemmatizer.pt',
    'pos.pretrain_path': 'saved_models/pos/en_ewt_tagger.pretrain.pt',
    'pos.model_path': 'saved_models/pos/en_ewt_tagger.pt',
    'depparse.pretrain_path': 'saved_models/depparse/en_ewt_parser.pretrain.pt',
    'depparse.model_path': 'saved_models/depparse/en_ewt_parser.pt'
}

def load_config(config_file_path):
    loaded_config = {}
    with open(config_file_path) as config_file:
        for config_line in config_file:
            config_key, config_value = config_line.split(':')
            loaded_config[config_key] = config_value.rstrip().lstrip()
    return loaded_config

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', help='language of text | default: en_ewt', default='en_ewt')
    parser.add_argument('-c', '--config', help='pipeline config file | default: None', default=None)
    parser.add_argument('text_file')
    args = parser.parse_args()
    # set up output file
    output_file_path = args.text_file+'.out'
    # set up pipeline
    if args.config is not None:
        print('loading pipeline configs from: '+args.config)
        pipeline_config = load_config(args.config)
    else:
        print('using default pipeline configs for: en_ewt')
        pipeline_config = english_config
    pipeline = Pipeline(config=pipeline_config)
    # run process
    # load input text
    input_text = open(args.text_file).read()
    # build document
    doc = Document(input_text)
    # run pipeline 
    pipeline.process(doc)
    # write conll to file
    doc.write_conll_to_file(output_file_path)

