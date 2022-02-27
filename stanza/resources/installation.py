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

DEFAULT_CORENLP_MODEL_URL = os.getenv(
    'CORENLP_MODEL_URL',
    'https://huggingface.co/stanfordnlp/corenlp-{model}/resolve/{tag}/stanford-corenlp-models-{model}.jar'
)
BACKUP_CORENLP_MODEL_URL = "http://nlp.stanford.edu/software/stanford-corenlp-{version}-models-{model}.jar"

DEFAULT_CORENLP_URL = os.getenv(
    'CORENLP_MODEL_URL',
    'https://huggingface.co/stanfordnlp/CoreNLP/resolve/{tag}/stanford-corenlp-latest.zip'
)

DEFAULT_CORENLP_DIR = os.getenv(
    'CORENLP_HOME',
    os.path.join(HOME_DIR, 'stanza_corenlp')
)

AVAILABLE_MODELS = set(['arabic', 'chinese', 'english-extra', 'english-kbp', 'french', 'german', 'hungarian', 'italian', 'spanish'])


def download_corenlp_models(model, version, dir=DEFAULT_CORENLP_DIR, url=DEFAULT_CORENLP_MODEL_URL, logging_level='INFO', proxies=None, force=True):
    """
    A automatic way to download the CoreNLP models.

    Args:
        model: the name of the model, can be one of 'arabic', 'chinese', 'english',
            'english-kbp', 'french', 'german', 'hungarian', 'italian', 'spanish'
        version: the version of the model
        dir: the directory to download CoreNLP model into; alternatively can be
            set up with environment variable $CORENLP_HOME
        url: The link to download CoreNLP models.
             It will need {model} and either {version} or {tag} to properly format the URL
        logging_level: logging level to use during installation
        force: Download model anyway, no matter model file exists or not
    """
    dir = os.path.expanduser(dir)
    if not model or not version:
        raise ValueError(
            "Both model and model version should be specified."
        )
    logger.info(f"Downloading {model} models (version {version}) into directory {dir}")
    model = model.strip().lower()
    if model not in AVAILABLE_MODELS:
        raise KeyError(
            f'{model} is currently not supported. '
            f'Must be one of: {list(AVAILABLE_MODELS)}.'
        )
    # for example:
    # https://huggingface.co/stanfordnlp/CoreNLP/resolve/v4.2.2/stanford-corenlp-models-french.jar
    tag = version if version == 'main' else 'v' + version
    download_url = url.format(tag=tag, model=model, version=version)
    model_path = os.path.join(dir, f'stanford-corenlp-{version}-models-{model}.jar')

    if os.path.exists(model_path) and not force:
        logger.warn(
            f"Model file {model_path} already exists. "
            f"Please download this model to a new directory.")
        return

    try:
        request_file(
            download_url,
            model_path,
            proxies
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        raise RuntimeError(
            "Downloading CoreNLP model file failed. "
            "Please try manual downloading at: https://stanfordnlp.github.io/CoreNLP/."
        ) from e


def install_corenlp(dir=DEFAULT_CORENLP_DIR, url=DEFAULT_CORENLP_URL, logging_level=None, proxies=None, version="main"):
    """
    A fully automatic way to install and setting up the CoreNLP library 
    to use the client functionality.

    Args:
        dir: the directory to download CoreNLP model into; alternatively can be
            set up with environment variable $CORENLP_HOME
        url: The link to download CoreNLP models
             Needs a {version} or {tag} parameter to specify the version
        logging_level: logging level to use during installation
    """
    dir = os.path.expanduser(dir)
    set_logging_level(logging_level=logging_level, verbose=None)
    if os.path.exists(dir) and len(os.listdir(dir)) > 0:
        logger.warn(
            f"Directory {dir} already exists. "
            f"Please install CoreNLP to a new directory.")
        return

    logger.info(f"Installing CoreNLP package into {dir}")
    # First download the URL package
    logger.debug(f"Download to destination file: {os.path.join(dir, 'corenlp.zip')}")
    tag = version if version == 'main' else 'v' + version
    url = url.format(version=version, tag=tag)
    try:
        request_file(url, os.path.join(dir, 'corenlp.zip'), proxies)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        raise RuntimeError(
            "Downloading CoreNLP zip file failed. "
            "Please try manual installation: https://stanfordnlp.github.io/CoreNLP/."
        ) from e

    # Unzip corenlp into dir
    logger.debug("Unzipping downloaded zip file...")
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

    # Warn user to set up env
    if dir != DEFAULT_CORENLP_DIR:
        logger.warning(
            f"For customized installation location, please set the `CORENLP_HOME` "
            f"environment variable to the location of the installation. "
            f"In Unix, this is done with `export CORENLP_HOME={dir}`.")

