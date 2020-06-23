"""
Functions for setting up the environments.
"""

import os
import logging
import zipfile
import shutil

from stanza.resources.common import HOME_DIR, request_file, unzip, \
    get_root_from_zipfile, set_logging_level

logger = logging.getLogger('stanza')

DEFAULT_CORENLP_URL = os.getenv(
    'CORENLP_URL',
    "http://nlp.stanford.edu/software/"
)
DEFAULT_CORENLP_DIR = os.getenv(
    'CORENLP_HOME',
    os.path.join(HOME_DIR, 'stanza_corenlp')
)

AVAILABLE_MODELS = set(['arabic', 'chinese', 'english', 'english-kbp', 'french', 'german', 'spanish'])

def install_corenlp(dir=DEFAULT_CORENLP_DIR, url=DEFAULT_CORENLP_URL, model=None, model_version=None, logging_level='INFO'):
    """
    A fully automatic way to install and setting up the CoreNLP library 
    to use the client functionality.

    Args:
        dir: the directory to install CoreNLP package into; alternatively can be
            set up with environment variable $STANZA_CORENLP_DIR
        set_corenlp_home: whether to point $CORENLP_HOME to the new directory
            at the end of installation; default to be True
        logging_level: logging level to use duing installation
    """
    dir = os.path.expanduser(dir)
    set_logging_level(logging_level=logging_level, verbose=None)
    if os.path.exists(dir):
        logger.warn(f"{dir} is already existed. Please specify a new directory.")
        return

    logger.info(f"Installing CoreNLP package into {dir}...")
    # First download the URL package
    logger.debug(f"Download to destination file: {os.path.join(dir, 'corenlp.zip')}")
    try:
        request_file(url + 'stanford-corenlp-latest.zip', os.path.join(dir, 'corenlp.zip'))
    except:
        raise Exception(
            "Downloading CoreNLP zip file failed. "
            "Please try manual installation: https://stanfordnlp.github.io/CoreNLP/."
        )

    # Unzip corenlp into dir
    logger.info("Unzipping downloaded zip file...")
    unzip(dir, 'corenlp.zip')

    # By default CoreNLP will be unzipped into a version-dependent folder, 
    # e.g., stanford-corenlp-4.0.0. We need some hack around that and move
    # files back into our designated folder
    logger.debug(f"Moving files into the designated folder at: {dir}")
    corenlp_dirname = get_root_from_zipfile(os.path.join(dir, 'corenlp.zip'))
    corenlp_dirname = os.path.join(dir, corenlp_dirname)
    for f in os.listdir(corenlp_dirname):
        shutil.move(os.path.join(corenlp_dirname, f), dir)

    # Remove original zip and folder
    logger.debug("Removing downloaded zip file...")
    os.remove(os.path.join(dir, 'corenlp.zip'))
    shutil.rmtree(corenlp_dirname)

    # Download model
    if model is not None and model_version is not None:
        model = model.strip().lower()
        if model not in AVAILABLE_MODELS:
            raise KeyError(f'{model} is currently not supported. All the supported models: {list(AVAILABLE_MODELS)}.')
        try:
            request_file(url + f'stanford-corenlp-{model_version}-models-{model}.jar', os.path.join(dir, f'stanford-corenlp-{model_version}-models-{model}.jar'))
        except:
            raise Exception(
                "Downloading CoreNLP model file failed. "
                "Please try manual downloading: https://stanfordnlp.github.io/CoreNLP/."
            )

    # Warn user to set up env
    if dir != DEFAULT_CORENLP_DIR:
        logger.warn(f"For customized downloading path, please set the `CORENLP_HOME` environment variable to the location of the folder: `export CORENLP_HOME={dir}`.")

