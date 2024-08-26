""" Version Number Resolver """

import re
from pathlib import Path

# read the version file, extracting the resources version
# to set the resources/package version, visit ../Cargo.toml
cargo = Path(__file__).parent.parent / "Cargo.toml"
with open(cargo, 'r') as vf:
    version_file_contents = vf.read()
    VERSION = re.compile('version ?= ?\"(.*)\"').search(version_file_contents).group(1)
    RESOURCES = re.compile('resources-version ?= ?\"(.*)\"').search(version_file_contents).group(1)

# this publicises it to all Python packages for reading
__version__ = VERSION
__resources_version__ = RESOURCES

