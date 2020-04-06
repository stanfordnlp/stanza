#!/bin/bash
# Setup basic prerequisites for running the tests.
# This script sets environment variables, so it needs to be sourced from the root directory, i.e., `source tests/setup_test.sh`.

test_dir=./stanza_test

mkdir -p $test_dir
mkdir -p $test_dir/in
mkdir -p $test_dir/out
mkdir -p $test_dir/scripts
cp tests/data/external_server.properties $test_dir/scripts
cp tests/data/example_french.json $test_dir/out

models_dir=$test_dir/models
mkdir -p $models_dir
python -c "import stanza; stanza.download(lang='en', dir='${models_dir}', logging_level='info')"
python -c "import stanza; stanza.download(lang='fr', dir='${models_dir}', logging_level='info')"
echo "Models downloaded to ${models_dir}."

export STANZA_TEST_HOME=$test_dir
echo "Test setup completed. Test home directory set to: ${STANZA_TEST_HOME}"
