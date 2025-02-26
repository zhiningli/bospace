import os
import logging
import logging.config

# Log format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] - %(message)s"

# Ensure log folders exist
def ensure_log_directories():
    for folder in ["logs/database", "logs/service", "logs/api", "logs/test", "logs/archive"]:
        os.makedirs(folder, exist_ok=True)

ensure_log_directories()

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG"
        },
        "file_database": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/database/database.log",
            "formatter": "default",
            "level": "INFO",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3
        },
        "file_service": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/service/service.log",
            "formatter": "default",
            "level": "DEBUG",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3
        },
        "file_api": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/api/api.log",
            "formatter": "default",
            "level": "INFO",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3
        },
        "file_test": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/test/test.log",  
            "formatter": "default",
            "level": "DEBUG",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3
        }
    },
    "loggers": {
        "database": {
            "handlers": ["console", "file_database"],
            "level": "INFO",
            "propagate": False
        },
        "service": {
            "handlers": ["console", "file_service"],
            "level": "INFO",
            "propagate": False
        },
        "api": {
            "handlers": ["console", "file_api"],
            "level": "INFO",
            "propagate": False
        },
        "test": {
            "handlers": ["console", "file_test"], 
            "level": "DEBUG",
            "propagate": False
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING"
    }
}

# Set logging level based on environment
environment = os.getenv("ENV", "dev")

if environment == "production":
    LOGGING_CONFIG["handlers"]["console"]["level"] = "WARNING"

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Log environment status
logging.getLogger("root").info(f"Logging initialized in '{environment}' mode.")
