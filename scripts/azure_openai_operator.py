import logging

import typer
from dotenv import load_dotenv

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Azure OpenAI operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def chat(
    query: str = typer.Option(
        "What is the weather like today?",
        "--query",
        "-q",
        help="Query to run with the Azure OpenAI chat model",
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
    response = AzureOpenAiWrapper().chat_model.invoke(
        input=query,
    )
    logger.debug(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        )
    )
    logger.info(f"Output: {response.content}")


@app.command()
def reasoning(
    query: str = typer.Option(
        "患者のデータから考えられる病名を診断してください。年齢： 55歳, 性別： 男性, 主訴： 激しい胸の痛み、息切れ, 検査データ： 心電図異常、トロポニン値上昇, 病歴： 高血圧、喫煙歴あり",  # noqa: E501
        "--query",
        "-q",
        help="Query to run with the Azure OpenAI reasoning model",
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
    response = AzureOpenAiWrapper().reasoning_model.invoke(
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
