#!/bin/bash
# Setup basic prerequisites for running the tests.
# This script sets environment variables, so it needs to be sourced from the root directory, i.e., `source stanza/tests/setup_test.sh`.

if hash python3 2>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

test_dir=./stanza_test

mkdir -p $test_dir
mkdir -p $test_dir/in
mkdir -p $test_dir/out
mkdir -p $test_dir/scripts
cp stanza/tests/data/external_server.properties $test_dir/scripts
cp stanza/tests/data/example_french.json $test_dir/out
cp stanza/tests/data/tiny_emb.* $test_dir/in

models_dir=$test_dir/models
mkdir -p $models_dir
$PYTHON -c "import stanza; stanza.download(lang='en', model_dir='${models_dir}', logging_level='info')" || echo "failed to download english model"
$PYTHON -c "import stanza; stanza.download(lang='fr', model_dir='${models_dir}', logging_level='info')" || echo "failed to download french model"
$PYTHON -c "import stanza; stanza.download(lang='zh', model_dir='${models_dir}', logging_level='info')" || echo "failed to download chinese model"
echo "Models downloaded to ${models_dir}."

export STANZA_TEST_HOME=$test_dir
echo "Test setup completed. Test home directory set to: ${STANZA_TEST_HOME}"
