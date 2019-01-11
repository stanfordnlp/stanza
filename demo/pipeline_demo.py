from datetime import datetime
from stanfordnlp.pipeline import Document, Pipeline

# example documents
english_doc = Document('Barack Obama was born in Hawaii.  He was elected president in 2008.')
french_doc = Document('Emmanuel Macron est né à Amiens. Il a été élu président en 2017.')

# example configs
english_config = {
    'processors': 'tokenize,pos,lemma,depparse',
    'tokenize.model_path': 'saved_models/tokenize/en_ewt_tokenizer.pt',
    'lemma.model_path': 'saved_models/lemma/en_ewt_lemmatizer.pt',
    'pos.pretrain_path': 'saved_models/pos/en_ewt_tagger.pretrain.pt',
    'pos.model_path': 'saved_models/pos/en_ewt_tagger.pt',
    'depparse.pretrain_path': 'saved_models/depparse/en_ewt_parser.pretrain.pt',
    'depparse.model_path': 'saved_models/depparse/en_ewt_parser.pt'
}

french_config = {
    'processors': 'tokenize,mwt,pos,lemma,depparse',
    'tokenize.model_path': 'saved_models/tokenize/fr_gsd_tokenizer.pt',
    'mwt.model_path': 'saved_models/mwt/fr_gsd_mwt_expander.pt',
    'lemma.model_path': 'saved_models/lemma/fr_gsd_lemmatizer.pt',
    'pos.pretrain_path': 'saved_models/pos/fr_gsd_tagger.pretrain.pt',
    'pos.model_path': 'saved_models/pos/fr_gsd_tagger.pt',
    'depparse.pretrain_path': 'saved_models/depparse/fr_gsd_parser.pretrain.pt',
    'depparse.model_path': 'saved_models/depparse/fr_gsd_parser.pt'
}

print('---')
print('load pipeline')
print('\tstart: '+str(datetime.now()))

# english example
english_pipeline = Pipeline(config=english_config)
english_pipeline.process(english_doc)

print('\tend: '+str(datetime.now()))

print('---')
print('english example')
print('---')
print('tokens of first English sentence')
for tok in english_doc.sentences[0].tokens:
    print(tok.word + '\t' + tok.lemma + '\t' + tok.pos)
print('---')
print('dependency parse of first English sentence')
for dep_edge in english_doc.sentences[0].dependencies:
    print((dep_edge[0].word, dep_edge[1], dep_edge[2].word))
print('---')
print('run on a second english document')
second_english_doc = Document('I am a sentence.')
english_pipeline.process(second_english_doc)
print('---')
print('tokens of second English document')
for tok in second_english_doc.sentences[0].tokens:
    print(tok.word + '\t' + tok.lemma + '\t' + tok.pos)
print('---')
print('dependency parse of second English document')
for dep_edge in english_doc.sentences[0].dependencies:
    print((dep_edge[0].word, dep_edge[1], dep_edge[2].word))

# french example
french_pipeline = Pipeline(config=french_config)
french_pipeline.process(french_doc)

print('---')
print('french example')
print('---')
print('tokens of first French sentence')
for tok in french_doc.sentences[0].tokens:
    print(tok.word + '\t' + tok.lemma + '\t' + tok.pos)
print('---')
print('dependency parse of first French sentence')
for dep_edge in french_doc.sentences[0].dependencies:
    print((dep_edge[0].word, dep_edge[1], dep_edge[2].word))
