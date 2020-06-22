"""
Functions for setting up the environments.
"""

import os
import logging
import zipfile
import shutil

from stanza.utils.resources import HOME_DIR, request_file, unzip, set_logging_level

logger = logging.getLogger('stanza')

CORENLP_ENV_NAME = 'CORENLP_HOME'
CORENLP_LATEST_URL = "http://nlp.stanford.edu/software/stanford-corenlp-latest.zip"
DEFAULT_CORENLP_DIR = os.getenv(
    'STANZA_CORENLP_DIR',
    os.path.join(HOME_DIR, 'stanza_corenlp')
)

def install_corenlp(dir=DEFAULT_CORENLP_DIR, set_corenlp_home=True, logging_level='INFO'):
    """
    A fully automatic way to install and setting up the CoreNLP library 
    to use the client functionality.

    Args:
        dir: the directory to install CoreNLP package into; alternatively can be
            set up with environment variable $STANZA_CORENLP_DIR
        set_corenlp_home: whether to point $CORENLP_HOME to the new directory
            at the end of installation; default to be True
    """
    dir = os.path.expanduser(dir)
    set_logging_level(logging_level=logging_level, verbose=None)
    logger.info(f"Installing CoreNLP package into {dir}...")
    # First download the URL package
    dest_file = os.path.join(dir, 'corenlp.zip')
    logger.debug(f"Download to destination file: {dest_file}")
    try:
        request_file(CORENLP_LATEST_URL, dest_file)
    except:
        raise Exception(
            "Downloading CoreNLP zip file failed. "
            "Please try manual installation: https://stanfordnlp.github.io/CoreNLP/."
        )

    # Unzip corenlp into dir
    logger.info("Unzipping downloaded zip file...")
    unzip_into(dest_file, dir)

    # By default CoreNLP will be unzipped into a version-dependent folder, 
    # e.g., stanford-corenlp-4.0.0. We need some hack around that and move
    # files back into our designated folder
    logger.debug(f"Moving files into the designated folder at: {dir}")
    corenlp_dirname = get_root_from_zipfile(dest_file)
    corenlp_dirname = os.path.join(dir, corenlp_dirname)
    for f in os.listdir(corenlp_dirname):
        shutil.move(os.path.join(corenlp_dirname, f), dir)

    # Remove original zip and folder
    logger.debug("Removing downloaded zip file...")
    os.remove(dest_file)
    shutil.rmtree(corenlp_dirname)

    # Set up env
    if set_corenlp_home:
        os.environ[CORENLP_ENV_NAME] = dir
        logger.info(f"Set environement variable {CORENLP_ENV_NAME} = {dir}")
    logger.info("CoreNLP installation completes.")

def unzip_into(filename, dest_dir):
    """
    Unzip a file into a destination folder.
    """
    logger.debug(f'Unzip {filename} into directory {dest_dir}...')
    with zipfile.ZipFile(filename) as f:
        f.extractall(dest_dir)

def get_root_from_zipfile(filename):
    """
    Get the root directory from a archived zip file.
    """
    try:
        zf = zipfile.ZipFile(filename, "r")
    except:
        raise Exception(f"Failed loading zip file at {filename}.")
    assert len(zf.filelist) > 0, \
        f"Zip file at f{filename} seems to be corrupted. Please check it."
    return os.path.dirname(zf.filelist[0].filename)
