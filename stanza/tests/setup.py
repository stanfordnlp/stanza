import glob
import logging
import os
import shutil
import stanza
from stanza.resources import installation
from stanza.tests import TEST_HOME_VAR, TEST_DIR_BASE_NAME

logger = logging.getLogger('stanza')

test_dir = os.getenv(TEST_HOME_VAR, None)
if not test_dir:
    test_dir = os.path.join(os.getcwd(), TEST_DIR_BASE_NAME)
    logger.info("STANZA_TEST_HOME not set.  Will assume $PWD/stanza_test = %s", test_dir)
    logger.info("To use a different directory, export or set STANZA_TEST_HOME=...")

in_dir = os.path.join(test_dir, "in")
out_dir = os.path.join(test_dir, "out")
scripts_dir = os.path.join(test_dir, "scripts")
models_dir=os.path.join(test_dir, "models")
corenlp_dir=os.path.join(test_dir, "corenlp_dir")

os.makedirs(test_dir, exist_ok=True)
os.makedirs(in_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
os.makedirs(scripts_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(corenlp_dir, exist_ok=True)

logger.info("COPYING FILES")

shutil.copy("stanza/tests/data/external_server.properties", scripts_dir)
shutil.copy("stanza/tests/data/example_french.json", out_dir)
for emb_file in glob.glob("stanza/tests/data/tiny_emb.*"):
    shutil.copy(emb_file, in_dir)

logger.info("DOWNLOADING MODELS")

stanza.download(lang='en', model_dir=models_dir, logging_level='info')
stanza.download(lang="en", model_dir=models_dir, package=None, processors={"ner":"ncbi_disease"})
stanza.download(lang='fr', model_dir=models_dir, logging_level='info')
stanza.download(lang='zh', model_dir=models_dir, logging_level='info')
stanza.download(lang='multilingual', model_dir=models_dir, logging_level='info')

logger.info("DOWNLOADING CORENLP")

installation.install_corenlp(dir=corenlp_dir)
installation.download_corenlp_models(model="french", version="main", dir=corenlp_dir)
installation.download_corenlp_models(model="german", version="main", dir=corenlp_dir)
installation.download_corenlp_models(model="italian", version="main", dir=corenlp_dir)
installation.download_corenlp_models(model="spanish", version="main", dir=corenlp_dir)

logger.info("Test setup completed.")
