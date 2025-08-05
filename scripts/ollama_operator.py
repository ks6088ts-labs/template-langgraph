import logging

import typer
from dotenv import load_dotenv

from template_langgraph.llms.ollamas import OllamaWrapper
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Ollama operator CLI",
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

    logger.info("Running...")
    chat_model = OllamaWrapper().chat_model
    response = chat_model.invoke(
        input=query,
    )
    logger.debug(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        )
    )
    logger.info(f"Output: {response.content}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
