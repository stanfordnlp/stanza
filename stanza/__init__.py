from stanza.pipeline.core import Pipeline
from stanza.models.common.doc import Document
from stanza.resources.common import download
from stanza.resources.installation import install_corenlp, download_corenlp_models
from stanza._version import __version__, __resources_version__

import logging
logger = logging.getLogger('stanza')

# if the client application hasn't set the log level, we set it
# ourselves to INFO
if logger.level == 0:
    logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)

# also, if the client hasn't added any handlers for this logger
# (or a default handler), we add a handler of our own
#
# client can later do
#   logger.removeHandler(stanza.log_handler)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
