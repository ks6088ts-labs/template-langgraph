import logging
from logging import getLogger

from template_python.loggers import get_logger

logger = getLogger(__name__)


def test_get_logger(caplog):
    """
    Test the get_logger function to ensure it returns a logger instance
    and prints a debug message correctly.
    """
    logger.info("[TEST] Running test_get_logger")
    with caplog.at_level(logging.DEBUG):
        test_logger = get_logger(__name__)
        test_logger.setLevel(logging.DEBUG)
        test_logger.debug(f"{__name__} logger initialized")

    assert test_logger.name == __name__
    assert f"{__name__} logger initialized" in caplog.text
    assert "DEBUG" in caplog.text
    assert "test_loggers.py" in caplog.text
