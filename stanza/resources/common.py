"""
Common utilities for Stanza resources.
"""

from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from contextlib import contextmanager
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import requests
import shutil
import tempfile
from typing import DefaultDict, Dict, List, NamedTuple, Optional, Tuple, TypedDict, TypeVar, Union
import zipfile

import huggingface_hub
from packaging import version
from platformdirs import user_cache_dir
from tqdm.auto import tqdm

from stanza.utils.helper_func import make_table
from stanza.pipeline._constants import (
    CONSTITUENCY,
    COREF,
    DEPPARSE,
    LANGID,
    LEMMA,
    MORPHSEG,
    MWT,
    NER,
    POS,
    SENTIMENT,
    TOKENIZE,
)
from stanza.pipeline.registry import PIPELINE_NAMES, PROCESSOR_VARIANTS
from stanza.resources.default_packages import PACKAGES
from stanza._version import __resources_version__

logger = logging.getLogger('stanza')

# set home dir for default
USER_CACHE_DIR = user_cache_dir('stanza', 'StanfordNLP', __resources_version__)
STANFORDNLP_RESOURCES_URL = 'https://nlp.stanford.edu/software/stanza/stanza-resources/'
STANZA_RESOURCES_GITHUB = 'https://raw.githubusercontent.com/stanfordnlp/stanza-resources/'
DEFAULT_RESOURCES_URL = os.getenv('STANZA_RESOURCES_URL', STANZA_RESOURCES_GITHUB + 'main')
DEFAULT_RESOURCES_VERSION = os.getenv(
    'STANZA_RESOURCES_VERSION',
    __resources_version__
)
DEFAULT_MODEL_URL = os.getenv('STANZA_MODEL_URL', 'default')
DEFAULT_MODEL_DIR = os.getenv(
    'STANZA_RESOURCES_DIR',
    os.path.join(USER_CACHE_DIR, 'resources')
)

PRETRAIN_NAMES = ("pretrain", "forward_charlm", "backward_charlm")

ProcessorName = str
ProcessorNames = Sequence[ProcessorName]
ProcessorPackage = str
ProcessorPackages = Sequence[ProcessorPackage]
ProcessorPackageMap = Mapping[
    ProcessorName,
    Union[ProcessorPackage, ProcessorPackages],
]
NormalizedProcessorPackageMap = Dict[
    ProcessorName,
    Union[ProcessorPackage, ProcessorPackages],
]
Processors = Union[ProcessorNames, ProcessorPackageMap]
Package = Union[ProcessorPackage, ProcessorPackageMap]
Proxies = Mapping[str, str]
DownloadEntry = List[str]
Downloads = List[DownloadEntry]
_Path = Union[str, os.PathLike[str]]
_HFUrlParts = Tuple[str, str, str]
_JSONValue = Union[
    None,
    bool,
    int,
    float,
    str,
    List["_JSONValue"],
    Dict[str, "_JSONValue"],
]
LanguageResources = Dict[str, _JSONValue]
Resources = Dict[str, _JSONValue]

class ResourcesFileNotFoundError(FileNotFoundError):
    def __init__(self, resources_filepath: _Path) -> None:
        super().__init__(f"Resources file not found at: {resources_filepath}  Try to download the model again.")
        self.resources_filepath = resources_filepath

class UnknownLanguageError(ValueError):
    def __init__(self, unknown: str) -> None:
        super().__init__(f"Unknown language requested: {unknown}")
        self.unknown_language = unknown

class UnknownProcessorError(ValueError):
    def __init__(self, unknown: str) -> None:
        super().__init__(f"Unknown processor type requested: {unknown}")
        self.unknown_processor = unknown

ModelDependency = Tuple[str, str]
ModelDependencies = Tuple[ModelDependency, ...]

class ModelSpecification(NamedTuple):
    processor: str
    package: str
    dependencies: Optional[ModelDependencies]

ModelSpecifications = Sequence[ModelSpecification]
MutableProcessorEntry = List[Union[ProcessorName, ModelSpecifications]]
MutableProcessorEntries = List[MutableProcessorEntry]
ImmutableProcessorEntry = Tuple[ProcessorName, ModelSpecifications]
ProcessorEntry = Sequence[Union[ProcessorName, ModelSpecifications]]
ProcessorEntries = Sequence[ProcessorEntry]
_ProcessorEntryT = TypeVar(
    "_ProcessorEntryT",
    MutableProcessorEntry,
    ImmutableProcessorEntry,
)

class ResourceDependency(TypedDict):
    model: ProcessorName
    package: ProcessorPackage

ResourceDependencies = List[ResourceDependency]

class ModelResource(TypedDict, total=False):
    md5: str
    alternate_md5: str
    dependencies: ResourceDependencies

ProcessorModels = Dict[ProcessorPackage, ModelResource]
PackageProcessors = Dict[ProcessorName, ProcessorPackage]
PackageDefinition = Dict[
    str,
    Union[ProcessorPackage, PackageProcessors],
]
Packages = Dict[ProcessorPackage, PackageDefinition]
_KNOWN_PROCESSOR_NAMES = (
    LANGID,
    TOKENIZE,
    MWT,
    POS,
    LEMMA,
    CONSTITUENCY,
    COREF,
    DEPPARSE,
    SENTIMENT,
    NER,
    MORPHSEG,
) + PRETRAIN_NAMES


def _is_registered_processor_name(name: str) -> bool:
    return name in _KNOWN_PROCESSOR_NAMES or name in PIPELINE_NAMES


def _is_language_resource(value: _JSONValue) -> bool:
    if not isinstance(value, dict):
        return False
    language_keys = ("alias", "lang_name", PACKAGES) + _KNOWN_PROCESSOR_NAMES
    return any(key in value for key in language_keys)


def _resource_value_error(location: str, expected: str) -> ValueError:
    return ValueError(
        f"Invalid resources JSON at {location}: expected {expected}"
    )

def _read_optional_string(
        values: LanguageResources,
        key: str,
        location: str,
    ) -> Optional[str]:
    if key not in values:
        return None
    value = values[key]
    if not isinstance(value, str):
        raise _resource_value_error(f"{location}.{key}", "a string")
    return value

def _read_resource_dependencies(
        value: _JSONValue,
        location: str,
    ) -> ResourceDependencies:
    if not isinstance(value, list):
        raise _resource_value_error(location, "a list of model dependencies")
    dependencies: ResourceDependencies = []
    for index, dependency in enumerate(value):
        dependency_location = f"{location}[{index}]"
        if not isinstance(dependency, dict):
            raise _resource_value_error(
                dependency_location,
                "an object with string model and package fields",
            )
        model = dependency.get("model")
        package = dependency.get("package")
        if not isinstance(model, str) or not isinstance(package, str):
            raise _resource_value_error(
                dependency_location,
                "an object with string model and package fields",
            )
        dependencies.append({"model": model, "package": package})
    return dependencies

def _read_model_resource(
        value: _JSONValue,
        location: str,
    ) -> ModelResource:
    if not isinstance(value, dict):
        raise _resource_value_error(location, "a model resource object")
    model_resource: ModelResource = {}
    if "md5" in value:
        md5 = value["md5"]
        if not isinstance(md5, str):
            raise _resource_value_error(f"{location}.md5", "a string")
        model_resource["md5"] = md5
    if "alternate_md5" in value:
        alternate_md5 = value["alternate_md5"]
        if not isinstance(alternate_md5, str):
            raise _resource_value_error(
                f"{location}.alternate_md5",
                "a string",
            )
        model_resource["alternate_md5"] = alternate_md5
    if "dependencies" in value:
        dependencies = value["dependencies"]
        model_resource["dependencies"] = _read_resource_dependencies(
            dependencies,
            f"{location}.dependencies",
        )
    return model_resource

def _read_processor_models(
        language_resources: LanguageResources,
        processor: ProcessorName,
        location: str,
    ) -> Optional[ProcessorModels]:
    if processor not in language_resources:
        return None
    value = language_resources[processor]
    if not isinstance(value, dict):
        raise _resource_value_error(
            f"{location}.{processor}",
            "an object mapping package names to model resources",
        )
    models: ProcessorModels = {}
    for package, model_resource in value.items():
        models[package] = _read_model_resource(
            model_resource,
            f"{location}.{processor}.{package}",
        )
    return models

def _read_package_processors(
        value: _JSONValue,
        location: str,
    ) -> PackageProcessors:
    if not isinstance(value, dict):
        raise _resource_value_error(
            location,
            "an object mapping processor names to packages",
        )
    processors: PackageProcessors = {}
    for processor, package in value.items():
        if not _is_registered_processor_name(processor):
            continue
        if not isinstance(package, str):
            raise _resource_value_error(
                f"{location}.{processor}",
                "a package name string",
            )
        processors[processor] = package
    return processors

def _read_package_definition(
        value: _JSONValue,
        location: str,
    ) -> PackageDefinition:
    if not isinstance(value, dict):
        raise _resource_value_error(location, "a package definition object")
    definition: PackageDefinition = {}
    for processor, package in value.items():
        if processor == "optional":
            definition[processor] = _read_package_processors(
                package,
                f"{location}.optional",
            )
        elif not _is_registered_processor_name(processor):
            continue
        elif not isinstance(package, str):
            raise _resource_value_error(
                f"{location}.{processor}",
                "a package name string",
            )
        else:
            definition[processor] = package
    return definition

def _read_packages(
        language_resources: LanguageResources,
        location: str,
    ) -> Packages:
    if PACKAGES not in language_resources:
        return {}
    value = language_resources[PACKAGES]
    if not isinstance(value, dict):
        raise _resource_value_error(
            f"{location}.{PACKAGES}",
            "an object mapping package names to package definitions",
        )
    packages: Packages = {}
    for package, definition in value.items():
        packages[package] = _read_package_definition(
            definition,
            f"{location}.{PACKAGES}.{package}",
        )
    return packages

def _validate_resources_json(value: _JSONValue) -> Resources:
    if not isinstance(value, dict):
        raise _resource_value_error("resources", "an object")
    if "url" in value and not isinstance(value["url"], str):
        raise _resource_value_error("resources.url", "a string")
    return value

def _require_language_resources(
        resources: Resources,
        lang: str,
    ) -> LanguageResources:
    language_resources = resources.get(lang)
    if not isinstance(language_resources, dict):
        raise _resource_value_error(
            f"resources.{lang}",
            "a language resource object",
        )
    return language_resources

def ensure_dir(path: _Path) -> None:
    """
    Create dir in case it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_md5(path: _Path) -> str:
    """
    Get the MD5 value of a path.
    """
    try:
        with open(path, 'rb') as fin:
            data = fin.read()
    except OSError as e:
        if not e.filename:
            e.filename = path
        raise
    return hashlib.md5(data).hexdigest()

def _is_within_directory(directory: _Path, target: _Path) -> bool:
    """
    Check that `target` resolves to a path inside `directory`.
    """
    directory = os.path.realpath(directory)
    target = os.path.realpath(target)
    return os.path.commonpath([directory]) == os.path.commonpath([directory, target])

def unzip(path: _Path, filename: _Path) -> None:
    """
    Fully unzip a file `filename` that's in a directory `dir`.

    Before unzipping, paths are checked so that a 'zip slip' error cannot happen.
    See https://github.com/stanfordnlp/stanza/security/advisories/GHSA-2fwf-f686-7p34
    """
    logger.debug(f'Unzip: {path}/{filename}...')
    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        for member in f.namelist():
            member_path = os.path.join(path, member)
            if not _is_within_directory(path, member_path):
                raise ValueError(
                    f"Zip file {filename} contains an entry that would extract "
                    f"outside of the target directory: {member}"
                )
        f.extractall(path)

def get_root_from_zipfile(filename: _Path) -> str:
    """
    Get the root directory from a archived zip file.
    """
    zf = zipfile.ZipFile(filename, "r")
    assert len(zf.filelist) > 0, \
        f"Zip file at f{filename} seems to be corrupted. Please check it."
    return os.path.dirname(zf.filelist[0].filename)

def file_exists(path: _Path, md5: Optional[str]) -> bool:
    """
    Check if the file at `path` exists and match the provided md5 value.
    """
    return os.path.exists(path) and get_md5(path) == md5

def assert_file_exists(path: _Path, md5: Optional[str] = None,
                       alternate_md5: Optional[str] = None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, "Cannot find expected file", path)
    if md5:
        file_md5 = get_md5(path)
        if file_md5 != md5:
            if file_md5 == alternate_md5:
                logger.debug("Found a possibly older version of file %s, md5 %s instead of %s", path, alternate_md5, md5)
            else:
                raise ValueError("md5 for %s is %s, expected %s" % (path, file_md5, md5))

_HF_URL_RE = re.compile(
    r'^https://huggingface\.co/(?P<repo_id>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/(?P<filename>.+)$'
)

def _parse_hf_url(url: str) -> Optional[_HFUrlParts]:
    m = _HF_URL_RE.match(url)
    if m is None:
        return None
    return m.group('repo_id'), m.group('revision'), m.group('filename')

def download_file(url: str, path: _Path, proxies: Optional[Proxies],
                  raise_for_status: bool = False) -> int:
    """
    Download a URL into a file as specified by `path`.

    For HuggingFace Hub URLs (when no proxy is configured), routes through
    huggingface_hub so that Xet, retries, and auth are handled correctly.
    Falls back to raw requests for non-HF URLs or when proxies are in use.
    """
    hf_parts = _parse_hf_url(url)
    if hf_parts is not None and not proxies:
        repo_id, revision, filename = hf_parts
        # TODO: here we could use local_dir, but the local layout
        #   en/constituency/ptb3-revised_electra-large.pt
        # unfortunately does not match the HF layout
        #   https://huggingface.co/stanfordnlp/stanza-en/resolve/v1.13.0/models/constituency/ptb3-revised_electra-large.pt
        if version.parse(huggingface_hub.__version__) < version.parse("1.0.0"):
            cached = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir_use_symlinks=False,
            )
        else:
            cached = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        shutil.copy2(cached, path)
        return 200

    verbose = logger.level in [0, 10, 20]
    request_proxies = dict(proxies) if proxies is not None else None
    r = requests.get(url, stream=True, proxies=request_proxies)
    if raise_for_status:
        r.raise_for_status()
    with open(path, 'wb') as f:
        file_size_header = r.headers.get('content-length', None)
        file_size = int(file_size_header) if file_size_header else None
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True, \
            disable=not verbose, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))
    return r.status_code

def request_file(url: str, path: _Path, proxies: Optional[Proxies] = None,
                 md5: Optional[str] = None, raise_for_status: bool = False,
                 log_info: bool = True, alternate_md5: Optional[str] = None) -> None:
    """
    A complete wrapper over download_file() that also make sure the directory of
    `path` exists, and that a file matching the md5 value does not exist.

    alternate_md5 allows for an alternate md5 that is acceptable (such as if an older version of a file is okay)
    """
    basedir = Path(path).parent
    ensure_dir(basedir)
    if file_exists(path, md5):
        if log_info:
            logger.info(f'File exists: {path}')
        else:
            logger.debug(f'File exists: {path}')
        return
    # We write data first to a temporary directory,
    # then use os.replace() so that multiple processes
    # running at the same time don't clobber each other
    # with partially downloaded files
    # This was especially common with resources.json
    with tempfile.TemporaryDirectory(dir=basedir) as temp:
        temppath = os.path.join(temp, os.path.split(path)[-1])
        download_file(url, temppath, proxies, raise_for_status)
        os.replace(temppath, path)
    assert_file_exists(path, md5, alternate_md5)
    if log_info:
        logger.info(f'Downloaded file to {path}')
    else:
        logger.debug(f'Downloaded file to {path}')

def sort_processors(
        processor_list: Sequence[_ProcessorEntryT],
    ) -> List[_ProcessorEntryT]:
    sorted_list: List[_ProcessorEntryT] = []
    for processor in PIPELINE_NAMES:
        for item in processor_list:
            if item[0] == processor:
                sorted_list.append(item)
    # going just by processors in PIPELINE_NAMES, this drops any names
    # which are not an official processor but might still be useful
    # check the list and append them to the end
    # this is especially useful when downloading pretrain or charlm models
    for processor in processor_list:
        for item in sorted_list:
            if processor[0] == item[0]:
                break
        else:
            sorted_list.append(processor)
    return sorted_list

def add_mwt(
        processors: NormalizedProcessorPackageMap,
        resources: Resources,
        lang: str,
    ) -> None:
    """Add mwt if tokenize is passed without mwt.

    If tokenize is in the list, but mwt is not, and there is a corresponding
    tokenize and mwt pair in the resources file, mwt is added so no missing
    mwt errors are raised.
    """
    if MWT in processors:
        return

    language_resources = _require_language_resources(resources, lang)
    resource_location = f"resources.{lang}"
    packages = _read_packages(language_resources, resource_location)
    tokenize_models = _read_processor_models(
        language_resources,
        TOKENIZE,
        resource_location,
    )
    mwt_models = _read_processor_models(
        language_resources,
        MWT,
        resource_location,
    )

    tokenize_packages = processors[TOKENIZE]
    if isinstance(tokenize_packages, str):
        tokenize_packages = (tokenize_packages,)
    elif not all(isinstance(package, str) for package in tokenize_packages):
        raise ValueError(
            "Processor values must be strings or sequences of strings"
        )

    mwt_packages: List[str] = []
    for tokenize_package in tokenize_packages:
        mwt_package: Optional[str] = None
        package_definition = packages.get(tokenize_package)
        if package_definition is not None and MWT in package_definition:
            mwt_package = tokenize_package
        elif (tokenize_models is not None
              and tokenize_package in tokenize_models
              and mwt_models is not None
              and tokenize_package in mwt_models):
            mwt_package = tokenize_package
        elif (tokenize_models is not None
              and tokenize_package in tokenize_models
              and mwt_models is not None):
            if tokenize_package.endswith("_nocharlm") or tokenize_package.endswith("_charlm"):
                base_package = tokenize_package.rsplit("_", maxsplit=1)[0]
                if base_package in mwt_models:
                    mwt_package = base_package

        if mwt_package is not None and mwt_package not in mwt_packages:
            logger.warning(
                "Language %s package %s expects mwt, which has been added",
                lang,
                mwt_package,
            )
            mwt_packages.append(mwt_package)

    if not mwt_packages:
        return
    if isinstance(processors[TOKENIZE], str):
        processors[MWT] = mwt_packages[0]
    else:
        processors[MWT] = mwt_packages

def maintain_processor_list(
        resources: Resources,
        lang: str,
        package: Optional[str],
        processors: Optional[NormalizedProcessorPackageMap],
        allow_pretrain: bool = False,
        maybe_add_mwt: bool = True,
    ) -> MutableProcessorEntries:
    """
    Given a parsed resources file, language, and possible package
    and/or processors, expands the package to the list of processors

    Returns a list of processors
    Each item in the list of processors is a pair:
      name, then a list of ModelSpecification
    so, for example:
      [['pos', [ModelSpecification(processor='pos', package='gsd', dependencies=None)]],
       ['depparse', [ModelSpecification(processor='depparse', package='gsd', dependencies=None)]]]
    """
    language_resources = _require_language_resources(resources, lang)
    resource_location = f"resources.{lang}"
    packages = _read_packages(language_resources, resource_location)
    processor_list: DefaultDict[str, List[str]] = defaultdict(list)
    # resolve processor models
    if processors:
        logger.debug(f'Processing parameter "processors"...')
        if maybe_add_mwt and TOKENIZE in processors and MWT not in processors:
            add_mwt(processors, resources, lang)
        for key, plist in processors.items():
            if not isinstance(key, str):
                raise ValueError("Processor names must be strings")
            if not isinstance(plist, (str, Sequence)):
                raise ValueError(
                    "Processor values must be strings or sequences of strings"
                )
            if isinstance(plist, str):
                plist = [plist]
            elif not all(isinstance(package, str) for package in plist):
                raise ValueError(
                    "Processor values must be strings or sequences of strings"
                )
            if key not in PIPELINE_NAMES:
                if not allow_pretrain or key not in PRETRAIN_NAMES:
                    raise UnknownProcessorError(key)
            processor_models = _read_processor_models(
                language_resources,
                key,
                resource_location,
            )
            for value in plist:
                optional_package: Optional[str] = None
                package_definition = packages.get(value)
                if package_definition is not None:
                    optional_processors = package_definition.get("optional")
                    if (isinstance(optional_processors, dict)
                            and key in optional_processors):
                        optional_package = optional_processors[key]
                # check if keys and values can be found
                if processor_models is not None and value in processor_models:
                    logger.debug(f'Found {key}: {value}.')
                    processor_list[key].append(value)
                # allow values to be default in some cases
                elif value in packages and key in packages[value]:
                    package_value = packages[value][key]
                    if not isinstance(package_value, str):
                        raise _resource_value_error(
                            f"{resource_location}.{PACKAGES}.{value}.{key}",
                            "a package name string",
                        )
                    logger.debug(
                        f'Found {key}: {package_value}.'
                    )
                    processor_list[key].append(package_value)
                # optional defaults will be activated if specifically turned on
                elif optional_package is not None:
                    logger.debug(f"Found {key}: {optional_package}.")
                    processor_list[key].append(optional_package)
                # allow processors to be set to variants that we didn't implement
                elif value in PROCESSOR_VARIANTS[key]:
                    logger.debug(
                        f'Found {key}: {value}. '
                        f'Using external {value} variant for the {key} processor.'
                    )
                    processor_list[key].append(value)
                # allow lemma to be set to "identity"
                elif key == LEMMA and value == 'identity':
                    logger.debug(
                        f'Found {key}: {value}. Using identity lemmatizer.'
                    )
                    processor_list[key].append(value)
                # not a processor in the officially supported processor list
                elif processor_models is None:
                    logger.debug(
                        f'{key}: {value} is not officially supported by Stanza, '
                        f'loading it anyway.'
                    )
                    processor_list[key].append(value)
                # cannot find the package for a processor and warn user
                else:
                    logger.warning(
                        f'Can not find {key}: {value} from official model list. '
                        f'Ignoring it.'
                    )
    # resolve package
    if package:
        logger.debug(f'Processing parameter "package"...')
        if package in packages:
            for key, value in packages[package].items():
                if key != 'optional' and key not in processor_list:
                    if not isinstance(value, str):
                        raise _resource_value_error(
                            f"{resource_location}.{PACKAGES}.{package}.{key}",
                            "a package name string",
                        )
                    logger.debug(f'Found {key}: {value}.')
                    processor_list[key].append(value)
        else:
            flag = False
            for key in PIPELINE_NAMES:
                processor_models = _read_processor_models(
                    language_resources,
                    key,
                    resource_location,
                )
                if processor_models is None:
                    continue
                if package in processor_models:
                    flag = True
                    if key not in processor_list:
                        logger.debug(f'Found {key}: {package}.')
                        processor_list[key].append(package)
                    else:
                        requested_processor = processors[key] if processors is not None else None
                        logger.debug(
                            f'{key}: {package} is overwritten by '
                            f'{key}: {requested_processor}.'
                        )
            if not flag: logger.warning((f'Can not find package: {package}.'))
    normalized_processor_list: MutableProcessorEntries = [
        [
            key,
            [
                ModelSpecification(
                    processor=key,
                    package=value,
                    dependencies=None,
                )
                for value in plist
            ],
        ]
        for key, plist in processor_list.items()
    ]
    return sort_processors(normalized_processor_list)

def add_dependencies(
        resources: Resources,
        lang: str,
        processor_list: MutableProcessorEntries,
    ) -> MutableProcessorEntries:
    """
    Expand the processor_list as given in maintain_processor_list to have the dependencies

    Still a list of model types to ModelSpecifications
    the dependencies are tuples: name and package
    for example:
    [['pos', (ModelSpecification(processor='pos', package='gsd', dependencies=(('pretrain', 'gsd'),)),)],
     ['depparse', (ModelSpecification(processor='depparse', package='gsd', dependencies=(('pretrain', 'gsd'),)),)]]
    """
    language_resources = _require_language_resources(resources, lang)
    resource_location = f"resources.{lang}"
    for item in processor_list:
        processor, model_specs = item
        if not isinstance(processor, str) or isinstance(model_specs, str):
            raise TypeError("Processor list entries must contain a name and model specifications")
        new_model_specs: List[ModelSpecification] = []
        for model_spec in model_specs:
            # skip dependency checking for external variants of processors and identity lemmatizer
            if not any([
                    model_spec.package in PROCESSOR_VARIANTS[processor],
                    processor == LEMMA and model_spec.package == 'identity'
                ]):
                processor_models = _read_processor_models(
                    language_resources,
                    processor,
                    resource_location,
                )
                model_resource = (
                    processor_models.get(model_spec.package)
                    if processor_models is not None
                    else None
                )
                resource_dependencies = (
                    model_resource.get("dependencies", [])
                    if model_resource is not None
                    else []
                )
                dependencies = [
                    (dependency["model"], dependency["package"])
                    for dependency in resource_dependencies
                ]
                model_spec = model_spec._replace(dependencies=tuple(dependencies))
                logger.debug("Found dependencies %s for processor %s model %s", dependencies, processor, model_spec.package)
            new_model_specs.append(model_spec)
        item[1] = tuple(new_model_specs)
    return processor_list

def flatten_processor_list(
        processor_list: ProcessorEntries,
    ) -> Downloads:
    """
    The flattened processor list is just a list of types & packages

    For example:
      [['pos', 'gsd'], ['depparse', 'gsd'], ['pretrain', 'gsd']]
    """
    flattened_processor_list: Downloads = []
    dependencies_list: List[ModelDependency] = []
    for item in processor_list:
        processor, model_specs = item
        if not isinstance(processor, str) or isinstance(model_specs, str):
            raise TypeError("Processor list entries must contain a name and model specifications")
        for model_spec in model_specs:
            package = model_spec.package
            dependencies = model_spec.dependencies
            flattened_processor_list.append([processor, package])
            if dependencies:
                dependencies_list.extend(dependencies)
    unique_dependencies: Downloads = [list(item) for item in set(dependencies_list)]
    for processor, package in unique_dependencies:
        logger.debug(f'Find dependency {processor}: {package}.')
    flattened_processor_list += unique_dependencies
    return flattened_processor_list

def set_logging_level(
        logging_level: Optional[str],
        verbose: Optional[bool],
    ) -> int:
    # Check verbose for easy logging control
    if verbose == False:
        logging_level = 'ERROR'
    elif verbose == True:
        logging_level = 'INFO'

    if logging_level is None:
        # default logging level of INFO is set in stanza.__init__
        # but the user may have set it via the logging API
        # it should NOT be 0, but let's check to be sure...
        if logger.level == 0:
            logger.setLevel('INFO')
        return logger.level

    # Set logging level
    logging_level = logging_level.upper()
    all_levels = ['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL', 'FATAL']
    if logging_level not in all_levels:
        raise ValueError(
            f"Unrecognized logging level for pipeline: "
            f"{logging_level}. Must be one of {', '.join(all_levels)}."
        )
    logger.setLevel(logging_level)
    return logger.level

@contextmanager
def logging_level_context(
        logging_level: Optional[str],
        verbose: Optional[bool],
    ) -> Iterator[None]:
    """
    Context manager that temporarily sets the stanza logger level for the
    duration of a managed block, then restores the original level on exit
    (including when the block raises an exception).

    If both arguments are None the logger is left completely untouched and
    no save/restore overhead is incurred.

    Example:
        with logging_level_context(logging_level, verbose):
            do_work()
        # logger level is back to whatever it was before do_work()
    """
    if logging_level is None and verbose is None:
        yield
        return

    original_level = logger.level
    try:
        set_logging_level(logging_level, verbose)
        yield
    finally:
        logger.setLevel(original_level)

def process_pipeline_parameters(
        lang: Optional[str],
        model_dir: Optional[str],
        package: Optional[Package],
        processors: Optional[Union[ProcessorName, Processors]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[NormalizedProcessorPackageMap]]:
    # Check parameter types and convert values to lower case
    if isinstance(lang, str):
        lang = lang.strip().lower()
    elif lang is not None:
        raise TypeError(
            f"The parameter 'lang' should be str, "
            f"but got {type(lang).__name__} instead."
        )

    if isinstance(model_dir, str):
        model_dir = model_dir.strip()
    elif model_dir is not None:
        raise TypeError(
            f"The parameter 'model_dir' should be str, "
            f"but got {type(model_dir).__name__} instead."
        )

    normalized_processors: Optional[NormalizedProcessorPackageMap]
    if isinstance(processors, (str, Sequence)):
        # Special case: processors is str, compatible with older version
        # also allow for setting alternate packages for these processors
        # via the package argument
        if package is None:
            # each processor will be 'default' for this language
            package = defaultdict(lambda: 'default')
        elif isinstance(package, str):
            # same, but now the named package will be the default instead
            default = package
            package = defaultdict(lambda: default)
        elif isinstance(package, Mapping):
            # the dictionary of packages will be used to build the processors dict
            # any processor not specified in package will be 'default'
            package = defaultdict(lambda: 'default', package)
        else:
            raise TypeError(
                f"The parameter 'package' should be None, str, or dict, "
                f"but got {type(package).__name__} instead."
            )
        if isinstance(processors, str):
            processor_names = [
                processor.strip().lower()
                for processor in processors.split(",")
            ]
        else:
            if not all(isinstance(processor, str) for processor in processors):
                raise TypeError(
                    "The parameter 'processors' must contain only strings"
                )
            processor_names = processors
        normalized_processors = {}
        for processor in processor_names:
            processor_package = package[processor]
            if not isinstance(processor_package, (str, Sequence)):
                raise TypeError(
                    "The parameter 'package' values must be strings "
                    "or sequences of strings"
                )
            if (not isinstance(processor_package, str)
                    and not all(isinstance(value, str) for value in processor_package)):
                raise TypeError(
                    "The parameter 'package' values must be strings "
                    "or sequences of strings"
                )
            normalized_processors[processor] = processor_package
        package = None
    elif isinstance(processors, Mapping):
        normalized_processors = {}
        for key, value in processors.items():
            if not isinstance(key, str):
                raise TypeError(
                    "The parameter 'processors' keys must be strings"
                )
            if isinstance(value, str):
                normalized_value: Union[ProcessorPackage, ProcessorPackages]
                normalized_value = value.strip().lower()
            elif isinstance(value, Sequence):
                if not all(isinstance(item, str) for item in value):
                    raise TypeError(
                        "The parameter 'processors' values must be strings "
                        "or sequences of strings"
                    )
                normalized_value = [item.strip().lower() for item in value]
            else:
                raise TypeError(
                    "The parameter 'processors' values must be strings "
                    "or sequences of strings"
                )
            normalized_processors[key.strip().lower()] = normalized_value
    elif processors is None:
        normalized_processors = None
    elif processors is not None:
        raise TypeError(
            f"The parameter 'processors' should be a dict, str, or sequence of str, "
            f"but got {type(processors).__name__} instead."
        )

    if isinstance(package, str):
        package = package.strip().lower()
    elif package is not None:
        raise TypeError(
            f"The parameter 'package' should be str, or a dict if 'processors' is a str, "
            f"but got {type(package).__name__} instead."
        )

    return lang, model_dir, package, normalized_processors

def download_resources_json(
        model_dir: _Path = DEFAULT_MODEL_DIR,
        resources_url: str = DEFAULT_RESOURCES_URL,
        resources_branch: Optional[str] = None,
        resources_version: str = DEFAULT_RESOURCES_VERSION,
        resources_filepath: Optional[_Path] = None,
        proxies: Optional[Proxies] = None,
    ) -> None:
    """
    Downloads resources.json to obtain latest packages.
    """
    if resources_url == DEFAULT_RESOURCES_URL and resources_branch is not None:
        resources_url = STANZA_RESOURCES_GITHUB + resources_branch
    # handle short name for resources urls; otherwise treat it as url
    if resources_url.lower() in ('stanford', 'stanfordnlp'):
        resources_url = STANFORDNLP_RESOURCES_URL
    resources_url = f'{resources_url}/resources_{resources_version}.json'
    logger.debug('Downloading resource file from %s', resources_url)
    if resources_filepath is None:
        resources_filepath = os.path.join(model_dir, 'resources.json')
    # make request
    request_file(
        resources_url,
        resources_filepath,
        proxies,
        raise_for_status=True
    )


def load_resources_json(
        model_dir: _Path = DEFAULT_MODEL_DIR,
        resources_filepath: Optional[_Path] = None,
    ) -> Resources:
    """
    Unpack the resources json file from the given model_dir
    """
    if resources_filepath is None:
        resources_filepath = os.path.join(model_dir, 'resources.json')
    if not os.path.exists(resources_filepath):
        raise ResourcesFileNotFoundError(resources_filepath)
    with open(resources_filepath, encoding="utf-8") as fin:
        resources_value: _JSONValue = json.load(fin)
    return _validate_resources_json(resources_value)

def resolve_language_resources(
        resources: Resources,
        lang: str,
    ) -> Tuple[str, Optional[LanguageResources]]:
    """
    Resolve aliases and return the canonical language code and its resources.
    """
    if lang not in resources:
        return lang, None

    language_resources = resources[lang]
    if not _is_language_resource(language_resources):
        return lang, None
    if not isinstance(language_resources, dict):
        raise RuntimeError("Language resource validation did not return a dict")

    visited = {lang}
    while True:
        alias = _read_optional_string(
            language_resources,
            "alias",
            f"resources.{lang}",
        )
        if alias is None:
            return lang, language_resources
        if alias in visited:
            raise ValueError(
                f"Invalid resources JSON: language alias cycle at {alias}"
            )
        visited.add(alias)
        lang = alias
        language_resources = _require_language_resources(resources, lang)

def get_language_resources(
        resources: Resources,
        lang: str,
    ) -> Optional[LanguageResources]:
    """
    Get the resources for a language, following aliases if needed.
    """
    return resolve_language_resources(resources, lang)[1]

def list_available_languages(
        model_dir: _Path = DEFAULT_MODEL_DIR,
        resources_url: str = DEFAULT_RESOURCES_URL,
        resources_branch: Optional[str] = None,
        resources_version: str = DEFAULT_RESOURCES_VERSION,
        proxies: Optional[Proxies] = None,
    ) -> List[str]:
    """
    List the non-alias languages in the resources file
    """
    download_resources_json(model_dir, resources_url, resources_branch, resources_version, resources_filepath=None, proxies=proxies)
    resources = load_resources_json(model_dir)
    # isinstance(str) is because of fields such as "url"
    # 'alias' is because we want to skip German, alias of de, for example
    languages = [
        lang
        for lang, language_resources in resources.items()
        if (_is_language_resource(language_resources)
            and isinstance(language_resources, dict)
            and _read_optional_string(
                language_resources,
                "alias",
                f"resources.{lang}",
            ) is None)
    ]
    languages = sorted(languages)
    return languages

def expand_model_url(resources: Resources, model_url: str) -> str:
    """
    Returns the url in the resources dict if model_url is default, or returns the model_url
    """
    if model_url.lower() != "default":
        return model_url
    resource_url = resources.get("url")
    if not isinstance(resource_url, str):
        raise _resource_value_error("resources.url", "a string")
    return resource_url

def download_models(
        download_list: Sequence[Sequence[str]],
        resources: Resources,
        lang: str,
        model_dir: _Path = DEFAULT_MODEL_DIR,
        resources_version: str = DEFAULT_RESOURCES_VERSION,
        model_url: str = DEFAULT_MODEL_URL,
        proxies: Optional[Proxies] = None,
        log_info: bool = True,
    ) -> None:
    language_resources = _require_language_resources(resources, lang)
    resource_location = f"resources.{lang}"
    lang_name_value = _read_optional_string(
        language_resources,
        "lang_name",
        resource_location,
    )
    lang_name = lang if lang_name_value is None else lang_name_value
    download_table = make_table(['Processor', 'Package'], download_list)
    if log_info:
        log_msg = logger.info
    else:
        log_msg = logger.debug
    log_msg(
        f'Downloading these customized packages for language: '
        f'{lang} ({lang_name})...\n{download_table}'
    )

    url = expand_model_url(resources, model_url)

    # Download packages
    for key, value in download_list:
        try:
            processor_models = _read_processor_models(
                language_resources,
                key,
                resource_location,
            )
            if processor_models is None or value not in processor_models:
                raise KeyError(value)
            model_resource = processor_models[value]
            md5 = model_resource.get("md5")
            if md5 is None:
                raise KeyError("md5")
            request_file(
                url.format(resources_version=resources_version, lang=lang, filename=f"{key}/{value}.pt"),
                os.path.join(model_dir, lang, key, f'{value}.pt'),
                proxies,
                md5=md5,
                log_info=log_info,
                alternate_md5=model_resource.get("alternate_md5"),
            )
        except KeyError as e:
            raise ValueError(
                f'Cannot find the following processor and model name combination: '
                f'{key}, {value}. Please check if you have provided the correct model name.'
            ) from e

# main download function
def download(
        lang: str = 'en',
        model_dir: str = DEFAULT_MODEL_DIR,
        package: Optional[Package] = 'default',
        processors: Optional[Union[ProcessorName, Processors]] = {},
        logging_level: Optional[str] = None,
        verbose: Optional[bool] = None,
        resources_url: str = DEFAULT_RESOURCES_URL,
        resources_branch: Optional[str] = None,
        resources_version: str = DEFAULT_RESOURCES_VERSION,
        model_url: str = DEFAULT_MODEL_URL,
        proxies: Optional[Proxies] = None,
        download_json: bool = True
    ) -> Downloads:

    # Temporarily adjust the log level for the duration of the download,
    # then restore it automatically.  If the caller passed neither argument
    # the logger is left completely untouched.
    # See https://github.com/stanfordnlp/stanza/issues/1418
    with logging_level_context(logging_level, verbose):
        # process different pipeline parameters
        normalized_lang, normalized_model_dir, package, processors = process_pipeline_parameters(
            lang, model_dir, package, processors
        )
        if normalized_lang is None or normalized_model_dir is None:
            raise ValueError("Download language and model directory cannot be None")
        lang = normalized_lang
        model_dir = normalized_model_dir
 
        if download_json or not os.path.exists(os.path.join(model_dir, 'resources.json')):
            if not download_json:
                logger.warning("Asked to skip downloading resources.json, but the file does not exist.  Downloading anyway")
            download_resources_json(model_dir, resources_url, resources_branch, resources_version, resources_filepath=None, proxies=proxies)

        resources = load_resources_json(model_dir)
        if lang not in resources:
            raise UnknownLanguageError(lang)
        requested_lang = lang
        lang, language_resources = resolve_language_resources(resources, lang)
        if language_resources is None:
            raise UnknownLanguageError(requested_lang)
        if lang != requested_lang:
            logger.info(f'"{requested_lang}" is an alias for "{lang}"')
        lang_name_value = _read_optional_string(
            language_resources,
            "lang_name",
            f"resources.{lang}",
        )
        lang_name = lang if lang_name_value is None else lang_name_value
        url = expand_model_url(resources, model_url)

        download_list: Downloads

        # Default: download zipfile and unzip
        if package == 'default' and (processors is None or len(processors) == 0):
            logger.info(
                f'Downloading default packages for language: {lang} ({lang_name}) ...'
            )
            default_md5 = _read_optional_string(
                language_resources,
                "default_md5",
                f"resources.{lang}",
            )
            if default_md5 is None:
                raise _resource_value_error(
                    f"resources.{lang}.default_md5",
                    "a string",
                )
            # want the URL to become, for example:
            # https://huggingface.co/stanfordnlp/stanza-af/resolve/v1.3.0/models/default.zip
            # so we hopefully start from
            # https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/v{resources_version}/models/{filename}
            request_file(
                url.format(resources_version=resources_version, lang=lang, filename="default.zip"),
                os.path.join(model_dir, lang, f'default.zip'),
                proxies,
                md5=default_md5,
            )
            unzip(os.path.join(model_dir, lang), 'default.zip')
            download_list = [['zip', 'default.zip']]
        # Customize: maintain download list
        else:
            processor_list = maintain_processor_list(resources, lang, package, processors, allow_pretrain=True)
            processor_list = add_dependencies(resources, lang, processor_list)
            download_list = flatten_processor_list(processor_list)
            download_models(download_list=download_list,
                            resources=resources,
                            lang=lang,
                            model_dir=model_dir,
                            resources_version=resources_version,
                            model_url=model_url,
                            proxies=proxies,
                            log_info=True)
        logger.info(f'Finished downloading models and saved to {model_dir}')
        return download_list
