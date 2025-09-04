import asyncio
import http.server
import json
import logging
import os
import socketserver
import tempfile
import webbrowser
from pathlib import Path
from urllib.parse import urljoin

# New imports for template rendering and serving
import jinja2
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


async def chat_impl() -> None:
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
    async with client.realtime.connect(
        model="gpt-realtime",  # name of your deployment
    ) as connection:
        await connection.session.update(
            session={
                "output_modalities": [
                    "text",
                    "audio",
                ],
                "model": "gpt-realtime",
                "type": "realtime",
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
                        {
                            "type": "input_text",
                            "text": user_input,
                        },
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
                    # logger.debug(f"event: {event.model_dump_json(indent=2)}")

    await credential.close()


@app.command()
def chat(
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

    asyncio.run(chat_impl())


@app.command()
def webrtc(
    template: str = typer.Option(
        "scripts/realtime_webrtc.html", "--template", "-t", help="Path to the HTML Jinja2 template"
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
    location: str = typer.Option("eastus2", "--location", help="location for Azure OpenAI"),
    deployment: str = typer.Option("gpt-realtime", "--deployment", help="Deployment name"),
    voice: str = typer.Option("verse", "--voice", help="Voice name"),
    instructions: str = typer.Option(
        "You are a helpful AI assistant responding in natural, engaging language.",
        "--instructions",
        "-i",
        help="Initial assistant instructions for the realtime session",
    ),
):
    """
    Render the realtime_webrtc HTML template with provided parameters and serve it as a static site.

    The template is a Jinja2 template and will receive the following variables:
    - WEBRTC_URL, SESSIONS_URL, API_KEY, DEPLOYMENT, VOICE, INSTRUCTIONS
    """
    settings = Settings()
    api_key = settings.azure_openai_api_key
    if not api_key:
        typer.secho(
            "Warning: no API key provided; the rendered page will contain an empty API key.", fg=typer.colors.YELLOW
        )

    tpl_path = Path(template)
    if not tpl_path.exists():
        typer.secho(f"Template not found: {tpl_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    tpl_text = tpl_path.read_text(encoding="utf-8")

    # Use json.dumps to safely embed JS string literals in the template
    rendered = jinja2.Template(tpl_text).render(
        WEBRTC_URL=json.dumps(f"https://{location}.realtimeapi-preview.ai.azure.com/v1/realtimertc"),
        SESSIONS_URL=json.dumps(
            urljoin(settings.azure_openai_endpoint, "/openai/realtimeapi/sessions?api-version=2025-04-01-preview")
        ),
        API_KEY=json.dumps(api_key),
        DEPLOYMENT=json.dumps(deployment),
        VOICE=json.dumps(voice),
        INSTRUCTIONS=json.dumps(instructions),
    )

    tempdir = tempfile.TemporaryDirectory()
    out_path = Path(tempdir.name) / "index.html"
    out_path.write_text(rendered, encoding="utf-8")

    # Serve the temporary directory with the rendered HTML
    os.chdir(tempdir.name)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        typer.secho(f"Serving rendered template at: {url}", fg=typer.colors.GREEN)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            typer.secho("Shutting down server...", fg=typer.colors.YELLOW)
        finally:
            tempdir.cleanup()


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
