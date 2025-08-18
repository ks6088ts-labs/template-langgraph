import logging

import typer
from dotenv import load_dotenv

from template_langgraph.llms.azure_ai_foundrys import AzureAiFoundryWrapper
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Azure AI Foundry operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def chat(
    query: str = typer.Option(
        "Hello",
        "--query",
        "-q",
        help="The query to send to the AI",
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

    logger.info("Running Azure AI Foundry chat...")
    # https://learn.microsoft.com/azure/ai-foundry/quickstarts/get-started-code?tabs=python&pivots=fdp-project
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    settings = AzureAiFoundryWrapper().settings

    project = AIProjectClient(
        endpoint=settings.azure_ai_foundry_inference_endpoint,
        credential=DefaultAzureCredential(),
    )
    models = project.get_openai_client(api_version=settings.azure_ai_foundry_inference_api_version)
    response = models.chat.completions.create(
        model=settings.azure_ai_foundry_inference_model_chat,
        messages=[
            {"role": "user", "content": query},
        ],
    )
    logger.info(response.choices[0].message.content)


@app.command()
def chat_langchain(
    query: str = typer.Option(
        "Hello",
        "--query",
        "-q",
        help="The query to send to the AI",
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

    logger.info("Running Azure AI Foundry chat...")
    # FIXME: impl


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
