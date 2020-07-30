---
layout: page
title: Download & Usage
keywords: stanza, biomedical and clinical model, download, usage
permalink: '/biomed_model_usage.html'
nav_order: 3
parent: Biomedical Models
toc: false
---

At a high level, you can download and use the Stanza biomedical models in the same way as for the general NLP models. We provide some simple examples here; for full details please refer to the [Stanza Usage page](usage).

You can download and initialize the syntactic analysis pipeline with an English language argument `en` and a simple package name keyword. The following command will download and initialize the CRAFT pipeline, annotate an example sentence, and print out the dependency structure:

```python
import stanza
# download and initialize the CRAFT pipeline
stanza.download('en', package='craft')
nlp = stanza.Pipeline('en', package='craft')
# annotate example text
doc = nlp('A single-cell transcriptomic atlas characterizes ageing tissues in the mouse.')
# print out dependency tree
doc.sentences[0].print_dependencies()
```

You can download and initialize the biomedical NER models by passing a dict with the `processors` argument. The following example downloads and initializes a pipeline with the MIMIC syntactic analysis models and the i2b2 clinical NER model, and print out all the annotated entities:

```python
# download and initialize a mimic pipeline with an i2b2 NER model
stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'})
# annotate clinical text
doc = nlp('The patient had a sore throat and was treated with Cepacol lozenges.')
# print out all entities
for ent in doc.entities:
    print(f'{ent.text}\t{ent.type}')
```

You can also choose to use the NER models alone. But in this case, since the text has to be pretokenized, you'll need to pass in a special `tokenize_pretokenized=True` argument, and you'll want to completely turn off the syntactic analysis pipeline with `package=None`:

```python
nlp = stanza.Pipeline('en', package=None, processors={'ner': 'i2b2'}, tokenize_pretokenized=True)
# annotate pre-tokenized sentences
doc = nlp([['He', 'had', 'a', 'sore', 'throat', '.'], ['He', 'was', 'treated', 'with', 'Cepacol', 'lozenges', '.']])
```