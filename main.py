"""Provide the command-line entry point for the Morocco Tourism Agent."""

import logging


LOGGER = logging.getLogger(__name__)
BANNER = """====================================
Morocco Tourism Agent
Agentic RAG using LangGraph
===================================="""


def main() -> None:
    """Configure logging and report that the project skeleton is ready."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info(BANNER)
    LOGGER.info("Project initialized successfully.")

    # TODO: Build and invoke the LangGraph workflow in a future stage.


if __name__ == "__main__":
    main()
