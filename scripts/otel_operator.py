import logging
import time

import typer
from dotenv import load_dotenv

from template_langgraph.internals.otel_helpers import OtelWrapper
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="OTEL operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def run(
    query: str = typer.Option(
        "What is the weather like today?",
        "--query",
        "-q",
        help="Query to run against the Ollama model",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    otel_wrapper = OtelWrapper()
    otel_wrapper.initialize()

    logger.info("Running...")
    tracer = otel_wrapper.get_tracer(name=__name__)
    with tracer.start_as_current_span("otel_operator_run"):
        logger.info(f"Query: {query}")
        # Simulate some work
        response = {"content": "It's sunny!"}
        time.sleep(1)  # Simulate processing time
        logger.info(f"Response: {response['content']}")


@app.command()
def cookbook(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    otel_wrapper = OtelWrapper()
    otel_wrapper.initialize()
    tracer = otel_wrapper.get_tracer(name=__name__)

    with tracer.start_as_current_span("otel_operator_cookbook"):
        logger.info("Running cookbook example...")
        time.sleep(1)

        with tracer.start_as_current_span("child"):
            logger.info("Running child span...")
            time.sleep(2)

        time.sleep(1)


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
