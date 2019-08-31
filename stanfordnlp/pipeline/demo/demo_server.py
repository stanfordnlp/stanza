from flask import Flask, request
import json
import stanfordnlp
app = Flask(__name__)

pipelineCache = dict()

@app.route('/', methods=['POST'])
def annotate():
    global pipelineCache

    properties = request.args.get('properties', '')
    lang = request.args.get('pipelineLanguage', '')
    text = list(request.form.keys())[0]

    if lang not in pipelineCache:
        pipelineCache[lang] = stanfordnlp.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma,depparse,ner', use_gpu=False)

    res = pipelineCache[lang](text)

    annotated_sentences = []
    for sentence in res.sentences:
        tokens = []
        deps = []
        for word in sentence.words:
            tokens.append({'index': word.id, 'word': word.text, 'lemma': word.lemma, 'pos': word.xpos, 'upos': word.upos, 'feats': word.feats, 'ner': word.ner if word.ner == 'O' else word.ner[2:]})
            deps.append({'dep': word.deprel, 'governor': word.head, 'governorGloss': sentence.words[word.head-1].text,
                'dependent': word.id, 'dependentGloss': word.text})
        annotated_sentences.append({'basicDependencies': deps, 'tokens': tokens})

    return json.dumps({'sentences': annotated_sentences})
