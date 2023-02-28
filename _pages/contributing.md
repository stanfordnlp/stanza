---
layout: page
title: Contributing
keywords: testing
permalink: '/contributing.html'
nav_order: 4
parent: Usage
---

## Git issues

If something looks wrong, or if you have a question, please fille out a [git issue](https://github.com/stanfordnlp/stanza/issues)

Many questions have already been asked and answered, so please do
search all issues, including closed issues, before filing a new issue.
Also, please upgrade to the latest version first.

## Pull requests

If you find yourself inspired to fix an issue or add a feature, you can send us
[a pull request](https://github.com/stanfordnlp/stanza/pulls)!
Please direct the pull request to the `dev` branch.

### Testing

Stanza has a large suite of unit tests which you can run as part of
the pull request.  Ideally, you would even add testing for a new
feature you add.

The tests use some models several times, along with a CoreNLP
distribution, so those pieces are downloaded in a setup script:

```python
python3 -m stanza.tests.setup
```

By default, the data objects will be downloaded to `stanza_test` in
whatever directory you run the setup script from.
This can be changed with the `TEST_HOME_VAR` environment variable.

The tests use `pytest`, which you may need to `pip install`.
After that, you can run them with `pytest -s stanza/tests`
in your git clone.
(You probably won't need to run the tests unless you are about to make a PR, anyway)

## New models and support for new languages

Generally speaking, we are happy to include new models in our
distributions.  Even better is when we have access to the data and can
rebuild the models in the future when the underlying data format or
the model structure changes.

The best thing in general is to file a new git issue, and we will
discuss the specific case in question.
