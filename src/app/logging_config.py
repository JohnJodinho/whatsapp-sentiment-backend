import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "sqlalchemy": {
            "level": "WARNING",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True
        },
        "sqlalchemy.engine": {
            "handlers": ["sqlalchemy"],
            "level": "WARNING",
            "propagate": False
        },
        "sqlalchemy.pool": {
            "handlers": ["sqlalchemy"],
            "level": "WARNING",
            "propagate": False
        }
    }
}