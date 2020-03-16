import logging.config
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "minimal": {
                "format": "%(message)s",
                }
            },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "minimal",
            }
        },
        "loggers": {
            "": {"handlers": ["console"], "level": "DEBUG"}
        },
    }
)
