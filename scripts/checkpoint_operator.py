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
    """List all available checkpoints with their thread IDs and basic information."""
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Using checkpoint type: {CHECKPOINT_LABELS.get(checkpoint_type, checkpoint_type)}")

    checkpointer = get_checkpointer(raw_value=checkpoint_type)
    if checkpointer is None:
        logger.info("No checkpointing is configured.")
        return

    try:
        checkpoints = list(checkpointer.list(config=None))

        if not checkpoints:
            logger.info("No checkpoints found.")
            return

        logger.info(f"Found {len(checkpoints)} checkpoint(s):")
        logger.info("-" * 60)

        for i, checkpoint in enumerate(checkpoints, 1):
            logger.debug(f"Checkpoint raw data: {checkpoint}")
            thread_id = checkpoint.config["configurable"].get("thread_id", "Unknown")
            checkpoint_id = checkpoint.config["configurable"].get("checkpoint_id", "Unknown")

            logger.info(f"{i}.")
            logger.info(f"   Thread ID: {thread_id}")
            logger.info(f"   Checkpoint ID: {checkpoint_id}")

            # Count messages in this checkpoint
            messages = checkpoint.checkpoint["channel_values"].get("messages") or []
            non_null_messages = [msg for msg in messages if msg is not None]
            logger.info(f"   Messages: {len(non_null_messages)} total")

            if verbose and non_null_messages:
                logger.info("   Recent messages:")
                # Show last 2 messages for brevity
                for msg in non_null_messages[-2:]:
                    if hasattr(msg, "content"):
                        content = str(msg.content)
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        msg_type = type(msg).__name__
                        logger.info(f"     [{msg_type}] {content_preview}")

            logger.info("-" * 60)

    except Exception as e:
        logger.error(f"Error listing checkpoints: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())


@app.command()
def list_messages(
    thread_id: str = typer.Option(
        ...,
        "--thread-id",
        "-i",
        help="Thread ID of the checkpoint to list messages from",
    ),
    checkpoint_type: str = typer.Option(
        DEFAULT_CHECKPOINT_TYPE.value,
        "--type",
        "-t",
        case_sensitive=False,
        help=f"Type of checkpoint to use. Options: {', '.join([f'{key} ({value})' for key, value in CHECKPOINT_LABELS.items()])}. Default is '{DEFAULT_CHECKPOINT_TYPE.value}'.",  # noqa: E501
    ),
    limit: int = typer.Option(None, "--limit", "-l", help="Maximum number of messages to display (default: all)"),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """List messages from a specific checkpoint thread."""
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Using checkpoint type: {CHECKPOINT_LABELS.get(checkpoint_type, checkpoint_type)}")
    logger.info(f"Retrieving messages for thread ID: {thread_id}")

    checkpointer = get_checkpointer(raw_value=checkpoint_type)
    if checkpointer is None:
        logger.info("No checkpointing is configured.")
        return

    try:
        # Search for the specific thread
        checkpoints = list(checkpointer.list(config=None))
        target_checkpoint = None

        for checkpoint in checkpoints:
            if checkpoint.config["configurable"].get("thread_id") == thread_id:
                target_checkpoint = checkpoint
                break

        if target_checkpoint is None:
            logger.error(f"Thread ID '{thread_id}' not found.")
            logger.info("Available thread IDs:")
            for checkpoint in checkpoints:
                available_thread_id = checkpoint.config["configurable"].get("thread_id")
                logger.info(f"  - {available_thread_id}")
            return

        # Extract messages
        messages = target_checkpoint.checkpoint["channel_values"].get("messages") or []
        non_null_messages = [msg for msg in messages if msg is not None]

        if not non_null_messages:
            logger.info("No messages found in this thread.")
            return

        # Apply limit if specified
        if limit and limit > 0:
            if limit < len(non_null_messages):
                logger.info(f"Showing last {limit} of {len(non_null_messages)} messages:")
                non_null_messages = non_null_messages[-limit:]
            else:
                logger.info(f"Showing all {len(non_null_messages)} messages:")
        else:
            logger.info(f"Showing all {len(non_null_messages)} messages:")

        logger.info("=" * 80)

        for i, msg in enumerate(non_null_messages, 1):
            msg_type = type(msg).__name__
            logger.info(f"Message {i} [{msg_type}]:")

            # Handle different message types
            if hasattr(msg, "content"):
                logger.info(f"  Content: {msg.content}")

            if hasattr(msg, "role"):
                logger.info(f"  Role: {msg.role}")

            if hasattr(msg, "name"):
                logger.info(f"  Name: {msg.name}")

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                logger.info(f"  Tool calls: {len(msg.tool_calls)}")
                if verbose:
                    for j, tool_call in enumerate(msg.tool_calls, 1):
                        logger.info(f"    {j}. {tool_call}")

            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs and verbose:
                logger.info(f"  Additional kwargs: {msg.additional_kwargs}")

            # Show raw message in verbose mode
            if verbose:
                logger.info(f"  Raw: {msg}")

            logger.info("-" * 40)

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
