import logging
from typing import Annotated

import typer
from dotenv import load_dotenv

from template_python.loggers import get_logger
from template_python.settings import Settings

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="template-python CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def hello(
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Name of the person to greet",
        ),
    ] = "World",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"This is a debug message with name: {name}")
    print(f"Hello {name}")
    settings = Settings()
    print(f"Settings from .env: {settings.model_dump_json(indent=2)}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
