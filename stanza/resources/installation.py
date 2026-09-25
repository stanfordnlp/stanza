"""
Functions for setting up the environments.
"""

import os
import logging
import re
import shutil

import huggingface_hub
import huggingface_hub.constants
import requests
from packaging import version as pkg_version

from stanza.resources.common import USER_CACHE_DIR, request_file, unzip, get_root_from_zipfile, logging_level_context, _parse_hf_url

logger = logging.getLogger('stanza')

DEFAULT_CORENLP_MODEL_URL = os.getenv(
    'CORENLP_MODEL_URL',
    'https://huggingface.co/stanfordnlp/corenlp-{model}/resolve/{tag}/stanford-corenlp-models-{model}.jar'
)
BACKUP_CORENLP_MODEL_URL = "http://nlp.stanford.edu/software/stanford-corenlp-{version}-models-{model}.jar"

DEFAULT_CORENLP_URL = os.getenv(
    'CORENLP_URL',
    'https://huggingface.co/stanfordnlp/CoreNLP/resolve/{tag}/stanford-corenlp-latest.zip'
)

DEFAULT_CORENLP_DIR = os.getenv(
    'CORENLP_HOME',
    os.path.join(USER_CACHE_DIR, 'corenlp')
)

AVAILABLE_MODELS = set(['arabic', 'chinese', 'english-extra', 'english-kbp', 'french', 'german', 'hungarian', 'italian', 'spanish'])

# the main CoreNLP jar in an installation, such as stanford-corenlp-4.5.10.jar
# the models, sources and javadoc jars do not match
CORENLP_JAR_RE = re.compile(r"stanford-corenlp-(\d+(?:\.\d+)+)\.jar")
# the top level directory of the CoreNLP zip, such as stanford-corenlp-4.5.10
CORENLP_ZIP_ROOT_RE = re.compile(r"stanford-corenlp-(\d+(?:\.\d+)+)")


def hf_repo_refs(repo_id, proxies=None):
    """
    Return ({branch: commit}, {tag: commit}) for a HuggingFace model repo

    Without proxies this goes through huggingface_hub, which respects
    HF_ENDPOINT and the other HF settings.  With a requests-style
    proxies dict it queries the same API endpoint through requests.
    """
    if proxies:
        endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        response = requests.get(f"{endpoint}/api/models/{repo_id}/refs", proxies=proxies, timeout=60)
        response.raise_for_status()
        data = response.json()
        branches = {ref["name"]: ref["targetCommit"] for ref in data["branches"]}
        tags = {ref["name"]: ref["targetCommit"] for ref in data["tags"]}
    else:
        refs = huggingface_hub.list_repo_refs(repo_id)
        branches = {ref.name: ref.target_commit for ref in refs.branches}
        tags = {ref.name: ref.target_commit for ref in refs.tags}
    return branches, tags


def cached_hf_refs(repo_id):
    """
    Names of the revisions of a HF model repo in the local HF cache, such as {"main", "v4.5.10"}
    """
    try:
        cache = huggingface_hub.scan_cache_dir()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.debug("Could not scan the HF cache: %s", e)
        return set()
    for repo in cache.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            return {ref for revision in repo.revisions for ref in revision.refs}
    return set()


def release_tags(tags):
    """
    Return {Version: tag name} for the tags which name a release, such as v4.5.10

    Other tags, and pre-releases, are ignored
    """
    releases = {}
    for name in tags:
        if not name.startswith("v"):
            continue
        try:
            parsed = pkg_version.Version(name[1:])
        except pkg_version.InvalidVersion:
            continue
        if parsed.is_prerelease:
            continue
        releases[parsed] = name
    return releases


def resolve_corenlp_version(url, requested, proxies=None):
    """
    Decide which revision of a CoreNLP download to fetch

    url is a download URL with any {model} already filled in.
    requested is one of:
      None     the newest release tagged in the HF repo
      "main"   the main branch, whatever it currently holds
      "4.5.10" (for example) that specific release

    Returns (version, tag, description):
      version     goes in local filenames: the release number, or "main"
      tag         the HF revision to download
      description is for log messages, such as "4.5.10" or "main (4.5.10)"
    """
    if requested is not None and requested != 'main':
        return requested, 'v' + requested, requested

    hf_parts = _parse_hf_url(url.format(tag='main', version='main'))
    # With HF_HUB_OFFLINE set, huggingface_hub can only serve revisions
    # already in its cache, so the newest release is the newest one there
    if hf_parts is not None and not proxies and huggingface_hub.constants.HF_HUB_OFFLINE:
        if requested == 'main':
            return 'main', 'main', 'main (offline)'
        repo_id = hf_parts[0]
        cached = cached_hf_refs(repo_id)
        releases = release_tags(cached)
        if releases:
            newest = max(releases)
            return str(newest), releases[newest], f"{newest} (newest in the offline cache)"
        if 'main' in cached:
            return 'main', 'main', 'main (offline)'
        raise RuntimeError(f"HF_HUB_OFFLINE is set, and {repo_id} is not in the HF cache")

    if requested is None:
        if hf_parts is None:
            raise ValueError(f"Cannot look up the newest CoreNLP version for {url}, "
                             "as it is not a HuggingFace URL.  Please give a version explicitly")
        repo_id = hf_parts[0]
        try:
            _, tags = hf_repo_refs(repo_id, proxies)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise RuntimeError(f"Could not look up the CoreNLP versions available in {repo_id}") from e
        releases = release_tags(tags)
        if not releases:
            raise RuntimeError(f"No release tags found in {repo_id}")
        newest = max(releases)
        return str(newest), releases[newest], str(newest)

    # requested == 'main'
    # The description says which release main matches.  This is only
    # informative, so a failure to look it up is not an error
    if hf_parts is None:
        return 'main', 'main', 'main'
    repo_id = hf_parts[0]
    try:
        branches, tags = hf_repo_refs(repo_id, proxies)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.debug("Could not look up the refs of %s: %s", repo_id, e)
        return 'main', 'main', 'main'
    main_commit = branches.get('main')
    releases = release_tags(tags)
    matching = [release for release, tag in releases.items() if tags[tag] == main_commit]
    if matching:
        description = f"main ({max(matching)})"
    else:
        description = f"main (untagged commit {main_commit[:10] if main_commit else 'unknown'})"
    if releases:
        newest = max(releases)
        if tags[releases[newest]] != main_commit:
            logger.warning("The main branch of %s is not the newest release, %s.  Downloading %s",
                           repo_id, newest, description)
    return 'main', 'main', description


def installed_corenlp_version(dir):
    """
    Return the version of the CoreNLP installed in dir, or None if there isn't exactly one
    """
    if not os.path.isdir(dir):
        return None
    versions = [match.group(1) for match in (CORENLP_JAR_RE.fullmatch(f) for f in os.listdir(dir)) if match]
    if len(versions) == 1:
        return versions[0]
    return None


def download_corenlp_models(model, version=None, dir=DEFAULT_CORENLP_DIR, url=DEFAULT_CORENLP_MODEL_URL, logging_level='INFO', proxies=None, force=True):
    """
    A automatic way to download the CoreNLP models.

    Args:
        model: the name of the model, can be one of 'arabic', 'chinese', 'english',
            'english-kbp', 'french', 'german', 'hungarian', 'italian', 'spanish'
        version: the version of the model, such as '4.5.10'.  'main' downloads
            whatever is on the main branch of the HF repo.  If None, the models
            match the CoreNLP installed in dir, or if there is no CoreNLP
            installed there, the newest release is used
        dir: the directory to download CoreNLP model into; alternatively can be
            set up with environment variable $CORENLP_HOME
        url: The link to download CoreNLP models.
             It will need {model} and either {version} or {tag} to properly format the URL
        logging_level: logging level to use during installation
        force: Download model anyway, no matter model file exists or not
    """
    dir = os.path.expanduser(dir)
    if not model:
        raise ValueError("The model to download must be specified")
    model = model.strip().lower()
    if model not in AVAILABLE_MODELS:
        raise KeyError(
            f'{model} is currently not supported. '
            f'Must be one of: {list(AVAILABLE_MODELS)}.'
        )

    if version is None:
        version = installed_corenlp_version(dir)
        if version is not None:
            logger.info(f"Using {model} models version {version} to match the CoreNLP installed in {dir}")
    url_for_model = url.replace('{model}', model)
    version, tag, description = resolve_corenlp_version(url_for_model, version, proxies)

    logger.info(f"Downloading {model} models {description} into directory {dir}")
    # for example:
    # https://huggingface.co/stanfordnlp/corenlp-french/resolve/v4.2.2/stanford-corenlp-models-french.jar
    download_url = url.format(tag=tag, model=model, version=version)
    model_path = os.path.join(dir, f'stanford-corenlp-{version}-models-{model}.jar')

    if os.path.exists(model_path) and not force:
        logger.warning(
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
            f"Downloading CoreNLP {model} models version {description} failed. "
            "Please try manual downloading at: https://stanfordnlp.github.io/CoreNLP/."
        ) from e


def install_corenlp(dir=DEFAULT_CORENLP_DIR, url=DEFAULT_CORENLP_URL, logging_level=None, proxies=None, version=None):
    """
    A fully automatic way to install and setting up the CoreNLP library 
    to use the client functionality.

    Args:
        dir: the directory to download CoreNLP model into; alternatively can be
            set up with environment variable $CORENLP_HOME
        url: The link to download CoreNLP models
             Needs a {version} or {tag} parameter to specify the version
        logging_level: logging level to use during installation
        proxies: requests-style proxies to use for the download
        version: the version of CoreNLP to install, such as '4.5.10'.
            'main' installs whatever is on the main branch of the HF repo.
            If None, the newest release is installed
    """
    dir = os.path.expanduser(dir)
    with logging_level_context(logging_level, verbose=None):
        if os.path.exists(dir) and len(os.listdir(dir)) > 0:
            existing = installed_corenlp_version(dir)
            existing = f" and contains CoreNLP {existing}" if existing else ""
            logger.warning(
                f"Directory {dir} already exists{existing}. "
                f"Please install CoreNLP to a new directory.")
            return

        version, tag, description = resolve_corenlp_version(url, version, proxies)
        logger.info(f"Installing CoreNLP {description} into {dir}")
        # First download the URL package
        logger.debug(f"Download to destination file: {os.path.join(dir, 'corenlp.zip')}")
        url = url.format(version=version, tag=tag)
        try:
            request_file(url, os.path.join(dir, 'corenlp.zip'), proxies)

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise RuntimeError(
                f"Downloading CoreNLP {description} failed. "
                "Please try manual installation: https://stanfordnlp.github.io/CoreNLP/."
            ) from e

        # Unzip corenlp into dir
        logger.debug("Unzipping downloaded zip file...")
        unzip(dir, 'corenlp.zip')

        # CoreNLP unzips into a version-dependent folder, such as
        # stanford-corenlp-4.0.0, so the files are moved from there
        # into the designated folder
        logger.debug(f"Moving files into the designated folder at: {dir}")
        corenlp_root = get_root_from_zipfile(os.path.join(dir, 'corenlp.zip'))
        corenlp_dirname = os.path.join(dir, corenlp_root)
        for f in os.listdir(corenlp_dirname):
            shutil.move(os.path.join(corenlp_dirname, f), dir)

        # Remove original zip and folder
        logger.debug("Removing downloaded zip file...")
        os.remove(os.path.join(dir, 'corenlp.zip'))
        shutil.rmtree(corenlp_dirname)

        # The folder name in the zip is the version actually installed
        root_match = CORENLP_ZIP_ROOT_RE.fullmatch(os.path.basename(os.path.normpath(corenlp_root)))
        if root_match:
            installed = root_match.group(1)
            logger.info(f"Installed CoreNLP {installed} into {dir}")
            if version != 'main' and installed != version:
                logger.warning(f"Requested CoreNLP {version}, but the download contained CoreNLP {installed}")
        else:
            logger.info(f"Installed CoreNLP {description} into {dir}")

        # Warn user to set up env
        if dir != DEFAULT_CORENLP_DIR:
            logger.warning(
                f"For customized installation location, please set the `CORENLP_HOME` "
                f"environment variable to the location of the installation. "
                f"In Unix, this is done with `export CORENLP_HOME={dir}`.")
