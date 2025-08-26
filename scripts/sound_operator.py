import logging

import sounddevice as sd
import soundfile as sf
import typer
from dotenv import load_dotenv

from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="agent runner CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def play(
    file: str = typer.Option(
        "input.wav",
        "--file",
        "-f",
        help="Path to the audio file to play",
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

    data, fs = sf.read(
        file=file,
        dtype="float32",
    )

    logger.info(f"Sampling rate: {fs}")
    logger.info(f"Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
    logger.info(f"Data type: {data.dtype}")
    logger.info(f"Duration: {len(data) / fs:.2f} seconds")

    sd.play(data, fs)
    sd.wait()


@app.command()
def list_devices(
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

    for idx, device in enumerate(sd.query_devices()):
        logger.info(f"Device {idx}:")
        for key, value in device.items():
            logger.info(f"  {key}: {value}")


@app.command()
def record(
    duration: float = typer.Option(
        5.0,
        "--duration",
        "-d",
        help="Duration to record audio (in seconds)",
    ),
    output: str = typer.Option(
        "output.wav",
        "--output",
        "-o",
        help="Path to the output audio file",
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

    logger.info(f"Recording audio for {duration} seconds...")
    samplerate = 44100
    channels = 2
    recording = sd.rec(
        frames=int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
    )
    sd.wait()
    sf.write(output, recording, samplerate)
    logger.info(f"Recording saved to {output}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
