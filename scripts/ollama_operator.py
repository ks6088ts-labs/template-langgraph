import logging
from base64 import b64encode
from logging import basicConfig

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from template_langgraph.llms.ollamas import OllamaWrapper, Settings
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


def load_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")


def set_verbose_logging(verbose: bool):
    if verbose:
        logger.setLevel(logging.DEBUG)
        basicConfig(level=logging.DEBUG)


@app.command()
def chat(
    query: str = typer.Option(
        "Explain the concept of Fourier transform.",
        "--query",
        "-q",
        help="Query to run against the Ollama model",
    ),
    model: str = typer.Option(
        "gemma3:270m",
        "--model",
        "-m",
        help="Model to use for structured output",
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
    set_verbose_logging(verbose)

    logger.info("Running...")
    chat_model = OllamaWrapper(
        settings=Settings(
            ollama_model_chat=model,
        ),
    ).chat_model

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
            response += str(chunk.content)
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
    model: str = typer.Option(
        "gemma3:270m",
        "--model",
        "-m",
        help="Model to use for structured output",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    set_verbose_logging(verbose)

    logger.info("Running...")
    chat_model = OllamaWrapper(
        settings=Settings(
            ollama_model_chat=model,
        ),
    ).chat_model
    profile = chat_model.with_structured_output(
        schema=Profile,
    ).invoke(
        input=[
            HumanMessage(content=query),
        ],
    )
    logger.info(f"Output: {profile.model_dump_json(indent=2, exclude_none=True)}")


@app.command()
def image(
    query: str = typer.Option(
        "Please analyze the following image and answer the question",
        "--query",
        "-q",
        help="Query to run with the image",
    ),
    file_path: str = typer.Option(
        "./docs/images/streamlit.png",
        "--file",
        "-f",
        help="Path to the image file to analyze",
    ),
    model: str = typer.Option(
        "gemma3:4b-it-q4_K_M",
        "--model",
        "-m",
        help="Model to use for image analysis",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    set_verbose_logging(verbose)

    base64_image = load_image_to_base64(file_path)
    messages = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": query,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    }

    logger.info("Running...")
    chat_model = OllamaWrapper(
        settings=Settings(
            ollama_model_chat=model,
        )
    ).chat_model
    response = chat_model.invoke(
        input=[
            messages,
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
def ocr(
    query: str = typer.Option(
        "Please extract all available details from the receipt image, including merchant/store name, transaction date (YYYY-MM-DD), total amount, and a fully itemized list (name, quantity, unit price, subtotal for each item).",  # noqa
        "--query",
        "-q",
        help="Query for OCR and comprehensive structured extraction from the receipt image",
    ),
    file_path: str = typer.Option(
        "./docs/images/streamlit.png",
        "--file",
        "-f",
        help="Path to the receipt image file for analysis",
    ),
    model: str = typer.Option(
        "gemma3:4b-it-q4_K_M",
        "--model",
        "-m",
        help="Model to use for OCR and structured information extraction",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    set_verbose_logging(verbose)
    from pydantic import BaseModel, Field

    class Item(BaseModel):
        item_name: str = Field(..., description="Exact name of the purchased item")
        quantity: int = Field(..., description="Number of units purchased")
        unit_price: float = Field(..., description="Unit price per item")
        total_price: float = Field(..., description="Subtotal for this item")

    class ReceiptInfo(BaseModel):
        merchant_name: str = Field(..., description="Full name of the merchant/store")
        transaction_date: str = Field(..., description="Transaction date in ISO format YYYY-MM-DD")
        total_amount: float = Field(..., description="Total amount paid, including tax")
        items: list[Item] = Field(
            ...,
            description="Detailed list of all purchased items with name, quantity, unit price, and subtotal",
        )

    base64_image = load_image_to_base64(file_path)
    messages = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": query,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    }

    logger.info("Running OCR and extracting detailed structured receipt information...")
    chat_model = OllamaWrapper(
        settings=Settings(
            ollama_model_chat=model,
        )
    ).chat_model
    response = chat_model.with_structured_output(ReceiptInfo).invoke(
        input=[
            messages,
        ],
    )
    logger.info(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        )
    )


@app.command()
def tool(
    query: str = typer.Option(
        "Please investigate troubleshooting cases for KABUTO.",
        "--query",
        "-q",
        help="Query for chat",
    ),
    model: str = typer.Option(
        "phi3:3.8b-mini-4k-instruct-q2_K",
        "--model",
        "-m",
        help="Model to use for Ollama",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    set_verbose_logging(verbose)

    from template_langgraph.tools.common import get_default_tools

    chat_model = OllamaWrapper(
        settings=Settings(
            ollama_model_chat=model,
        )
    ).chat_model
    response = chat_model.bind_tools(tools=get_default_tools()).invoke(
        input=[
            query,
        ],
    )
    logger.info(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        )
    )


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
