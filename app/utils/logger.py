"""Configure concise application-wide logging."""

import logging


QUIET_THIRD_PARTY_LOGGERS = (
    "google_genai.models",
    "httpcore",
    "httpx",
)


def configure_application_logging() -> None:
    """Enable application information while suppressing HTTP debug noise."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for logger_name in QUIET_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
