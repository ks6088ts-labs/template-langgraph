import asyncio
import logging

import typer
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from template_langgraph.llms.azure_openais import Settings
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Realtime API operator CLI",
)

# Set up logging
logger = get_logger(__name__)


async def main() -> None:
    """
    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    settings = Settings()
    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.azure_openai_api_version,
    )
    async with client.beta.realtime.connect(
        model="gpt-realtime",  # name of your deployment
    ) as connection:
        await connection.session.update(
            session={
                "output_modalities": [
                    "text",
                    "audio",
                ]
            }
        )
        while True:
            user_input = input("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_input},
                    ],
                }
            )
            await connection.response.create()
            async for event in connection:
                if event.type == "response.audio_transcript.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.done":
                    print()
                    break
                else:
                    logger.debug(f"event.type: {event.type}")

    await credential.close()


@app.command()
def run(
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

    asyncio.run(main())


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
