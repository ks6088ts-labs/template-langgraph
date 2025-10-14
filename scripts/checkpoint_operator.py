import logging
import os
import sqlite3
from enum import Enum

import typer
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_checkpoint_cosmosdb import CosmosDBSaver

from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Checkpoint Operator CLI",
)

# Set up logging
logger = get_logger(__name__)


class CheckpointType(str, Enum):
    SQLITE = "sqlite"
    COSMOSDB = "cosmosdb"
    MEMORY = "memory"
    NONE = "none"


DEFAULT_CHECKPOINT_TYPE = CheckpointType.NONE
CHECKPOINT_LABELS = {
    CheckpointType.COSMOSDB.value: "Cosmos DB",
    CheckpointType.SQLITE.value: "SQLite",
    CheckpointType.MEMORY.value: "メモリ",
    CheckpointType.NONE.value: "なし",
}


def get_selected_checkpoint_type(raw_value: str) -> CheckpointType:
    try:
        checkpoint = CheckpointType(raw_value)
    except ValueError:
        return DEFAULT_CHECKPOINT_TYPE
    return checkpoint


def get_checkpointer(raw_value: str):
    checkpoint_type = get_selected_checkpoint_type(
        raw_value=raw_value,
    )
    if checkpoint_type is CheckpointType.SQLITE:
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        return SqliteSaver(conn=conn)
    if checkpoint_type is CheckpointType.COSMOSDB:
        from template_langgraph.tools.cosmosdb_tool import get_cosmosdb_settings

        settings = get_cosmosdb_settings()
        os.environ["COSMOSDB_ENDPOINT"] = settings.cosmosdb_host
        os.environ["COSMOSDB_KEY"] = settings.cosmosdb_key

        return CosmosDBSaver(
            database_name=settings.cosmosdb_database_name,
            container_name="checkpoints",
        )
    if checkpoint_type is CheckpointType.MEMORY:
        return InMemorySaver()
    return None


@app.command()
def list_checkpoints(
    checkpoint_type: str = typer.Option(
        DEFAULT_CHECKPOINT_TYPE.value,
        "--type",
        "-t",
        case_sensitive=False,
        help=f"Type of checkpoint to list. Options: {', '.join([f'{key} ({value})' for key, value in CHECKPOINT_LABELS.items()])}. Default is '{DEFAULT_CHECKPOINT_TYPE.value}'.",  # noqa: E501
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

    checkpointer = get_checkpointer(raw_value=checkpoint_type)
    if checkpointer is None:
        logger.info("No checkpointing is configured.")
        return
    checkpoints = checkpointer.list(
        config=None,
    )
    for checkpoint in checkpoints:
        logger.info(f"Thread ID: {checkpoint.config['configurable'].get('thread_id')}")
        logger.info(f"{checkpoint.checkpoint['channel_values']}")
        messages = checkpoint.checkpoint["channel_values"].get("messages") or []
        for message in messages:
            if message is not None:
                logger.info(f"  - {message}")
            else:
                logger.info("  - None")


@app.command()
def list_messages(
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


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
