"""
Test installation functions.
"""

import os
import pytest
import shutil
import tempfile

import stanza

pytestmark = [pytest.mark.travis, pytest.mark.client]

def test_install_corenlp():
    # we do not reset the CORENLP_HOME variable since this may impact the 
    # client tests
    with tempfile.TemporaryDirectory(dir=".") as test_dir:

        # the download method doesn't install over existing directories
        shutil.rmtree(test_dir)
        stanza.install_corenlp(dir=test_dir, url='http://nlp.stanford.edu/software/')

        assert os.path.isdir(test_dir), "Installation destination directory not found."
        jar_files = [f for f in os.listdir(test_dir) \
                     if f.endswith('.jar') and f.startswith('stanford-corenlp')]
        assert len(jar_files) > 0, \
            "Cannot find stanford-corenlp jar files in the installation directory."
        assert not os.path.exists(os.path.join(test_dir, 'corenlp.zip')), \
            "Downloaded zip file was not removed."
    
def test_download_corenlp_models():
    model_name = "arabic"
    version = "4.2.2"

    with tempfile.TemporaryDirectory(dir=".") as test_dir:
        stanza.download_corenlp_models(model=model_name, version=version, dir=test_dir)

        dest_file = os.path.join(test_dir, f"stanford-corenlp-{version}-models-{model_name}.jar")
        assert os.path.isfile(dest_file), "Downloaded model file not found."
