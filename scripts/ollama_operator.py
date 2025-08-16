import logging

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from template_langgraph.llms.ollamas import OllamaWrapper
from template_langgraph.loggers import get_logger


class Profile(BaseModel):
    first_name: str = Field(..., description="First name of the user")
    last_name: str = Field(..., description="Last name of the user")
    age: int = Field(..., description="Age of the user")
    origin: str = Field(
        ...,
        description="Origin of the user, e.g., country or city",
    )


# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Ollama operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def chat(
    query: str = typer.Option(
        "Explain the concept of Fourier transform.",
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
    stream: bool = typer.Option(
        False,
        "--stream",
        "-s",
        help="Enable streaming output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Running...")
    chat_model = OllamaWrapper().chat_model

    if stream:
        response = ""
        for chunk in chat_model.stream(
            input=[
                HumanMessage(content=query),
            ],
        ):
            print(
                chunk.content,
                end="",
                flush=True,
            )
            response += chunk.content
        logger.info(f"Output: {response}")
    else:
        response = chat_model.invoke(
            input=[
                HumanMessage(content=query),
            ],
        )
        logger.debug(
            response.model_dump_json(
                indent=2,
                exclude_none=True,
            )
        )
        logger.info(f"Output: {response.content}")


@app.command()
def structured_output(
    query: str = typer.Option(
        "I'm Taro Okamoto from Japan. 30 years old.",
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
    profile = chat_model.with_structured_output(
        schema=Profile,
    ).invoke(
        input=[
            HumanMessage(content=query),
        ],
    )
    logger.info(f"Output: {profile.model_dump_json(indent=2, exclude_none=True)}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
