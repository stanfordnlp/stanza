from flask import Flask, request, abort
import json
import stanza
import os
app = Flask(__name__, static_url_path='', static_folder=os.path.abspath(os.path.dirname(__file__)))

pipelineCache = dict()

def get_file(path):
    res = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    print(res)
    return res

@app.route('/<path:path>')
def static_file(path):
    if path in ['stanza-brat.css', 'stanza-brat.js', 'stanza-parseviewer.js', 'loading.gif',
            'favicon.png', 'stanza-logo.png']:
        return app.send_static_file(path)
    elif path in 'index.html':
        return app.send_static_file('stanza-brat.html')
    else:
        abort(403)

@app.route('/', methods=['GET'])
def index():
    return static_file('index.html')

@app.route('/', methods=['POST'])
def annotate():
    global pipelineCache

    properties = request.args.get('properties', '')
    lang = request.args.get('pipelineLanguage', '')
    text = list(request.form.keys())[0]

    if lang not in pipelineCache:
        pipelineCache[lang] = stanza.Pipeline(lang=lang, use_gpu=False)

    res = pipelineCache[lang](text)

    annotated_sentences = []
    for sentence in res.sentences:
        tokens = []
        deps = []
        for word in sentence.words:
            tokens.append({'index': word.id, 'word': word.text, 'lemma': word.lemma, 'pos': word.xpos, 'upos': word.upos, 'feats': word.feats, 'ner': word.parent.ner if word.parent.ner is None or word.parent.ner == 'O' else word.parent.ner[2:]})
            deps.append({'dep': word.deprel, 'governor': word.head, 'governorGloss': sentence.words[word.head-1].text,
                'dependent': word.id, 'dependentGloss': word.text})
        annotated_sentences.append({'basicDependencies': deps, 'tokens': tokens})

    return json.dumps({'sentences': annotated_sentences})

def create_app():
    return app
