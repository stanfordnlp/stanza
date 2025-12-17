---
layout: default
title: Models
permalink: '/models.html'
nav_order: 4
has_children: true
---

# Models

In this section, we cover the list of supported human languages and models that are available for download in Stanza, the performance of these models, as well as how you can contribute models you trained to the Stanza community.

## Packages

Packages for download can be specified with the [Pipeline](pipeline.md).  The default packages include the tokenizer, mwt if appropriate, POS, lemma, dependencies, NER, sentiment, and constituencies, depending on availability.  The `default_accurate` package adds coref and uses transformers if we have trained transformer based models for that language.

Other packages available depend on the treebank used to train the
models, with recent results listed [on the performance
page](performance.md).  When multiple UD treebanks had compatible
annotation schemes, the default package uses
[combined models](combined_models.md).

{: .fs-5 .fw-300 }
