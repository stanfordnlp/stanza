#!/bin/bash
# Setup basic prerequisites for running the tests.
# This script needs to be called from the root directory, i.e., `bash tests/setup_test.sh`.

test_dir=./stanfordnlp_test

mkdir $test_dir
mkdir $test_dir/in
mkdir $test_dir/out
mkdir $test_dir/scripts
cp tests/data/external_server.properties $test_dir/scripts
cp tests/data/example_french.json $test_dir/out

export STANFORDNLP_TEST_HOME=$test_dir

