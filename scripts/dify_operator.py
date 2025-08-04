import json
import logging

import typer
from dotenv import load_dotenv

from template_langgraph.loggers import get_logger
from template_langgraph.tools.dify_tool import DifyClientWrapper

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Dify operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def run_workflow(
    requirements: str = typer.Option(
        "生成 AI のサービス概要を教えてください。日本語でお願いします",
        "--requirements",
        "-r",
        help="Requirements for running the Dify workflow",
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

    logger.info("Running Dify workflow...")
    client = DifyClientWrapper()
    response = client.run_workflow(
        inputs={
            "inputs": {
                "requirements": requirements,
            },
            "response_mode": "blocking",
            "user": "abc-123",
        }
    )
    logger.info(
        json.dumps(
            response,
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info(f"Input: {response['data']['outputs']['requirements']}, Output: {response['data']['outputs']['text']}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
