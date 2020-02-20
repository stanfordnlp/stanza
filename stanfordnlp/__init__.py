from stanfordnlp.pipeline.core import Pipeline
from stanfordnlp.models.common.doc import Document
from stanfordnlp.utils.resources import download
from stanfordnlp._version import __version__, __resources_version__

import logging.config
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)s: %(message)s",
                'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "loggers": {
            "": {"handlers": ["console"]}
        },
    }
)
