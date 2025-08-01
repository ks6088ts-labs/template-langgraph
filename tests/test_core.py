import logging
from logging import getLogger

import pytest

from template_python.core import hello_world

logger = getLogger(__name__)


@pytest.fixture(scope="session")
def setup_session():
    """
    Session-wide setup fixture that initializes resources
    needed for the tests. This fixture runs once per test session.
    It can be used to set up resources that are shared across multiple tests.
    """
    logger.info("[SETUP] Starting session-wide setup")
    resource = {"key": "value"}
    yield resource
    logger.info("[TEARDOWN] Cleaning up session-wide resources")


def test_hello_world_verbose(setup_session, caplog):
    """
    Test the hello_world function with verbose output.
    This test checks if the function logs the expected message
    and if the session setup is correctly initialized.
    """
    logger.info("[TEST] Running test_hello_world_verbose")
    with caplog.at_level(logging.DEBUG):
        hello_world(verbose=True)
    assert "Hello World" in caplog.text
    assert "key" in setup_session
    assert setup_session["key"] == "value"


def test_hello_world_non_verbose(setup_session, caplog):
    """
    Test the hello_world function without verbose output.
    This test checks if the function does not log the message
    when verbose is set to False, while still ensuring session setup.
    """
    logger.info("[TEST] Running test_hello_world_non_verbose")
    with caplog.at_level(logging.INFO):
        hello_world(verbose=False)
    assert "Hello World" not in caplog.text
    assert "key" in setup_session
    assert setup_session["key"] == "value"


@pytest.mark.parametrize(
    "verbose, expected_log",
    [
        (True, "Hello World"),
        (False, ""),
    ],
)
def test_hello_world_parametrized(setup_session, caplog, verbose, expected_log):
    """
    Parametrized test for the hello_world function.
    This test runs the function with different verbosity levels
    and checks if the expected log message is present.
    """
    logger.info(f"[TEST] Running test_hello_world_parametrized with verbose={verbose}")
    with caplog.at_level(logging.DEBUG if verbose else logging.INFO):
        hello_world(verbose=verbose)
    assert expected_log in caplog.text
    assert "key" in setup_session
    assert setup_session["key"] == "value"
