import logging

import typer
from dotenv import load_dotenv

from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="template-langgraph CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def png(
    name: str = typer.Option(
        "chat_with_tools_agent",
        "--name",
        "-n",
        help="Name of the agent to draw",
    ),
    output_file_path: str = typer.Option(
        "output.png",
        "--output",
        "-o",
        help="Path to the output PNG file",
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

    logger.debug(f"This is a debug message with name: {name}")
    if name == "chat_with_tools_agent":
        from template_langgraph.agents.chat_with_tools_agent.agent import graph
    if name == "issue_formatter_agent":
        from template_langgraph.agents.issue_formatter_agent.agent import graph
    if name == "task_decomposer_agent":
        from template_langgraph.agents.task_decomposer_agent.agent import graph
    if name == "kabuto_helpdesk_agent":
        from template_langgraph.agents.kabuto_helpdesk_agent import graph

    typer.echo(f"Drawing agent: {name}")
    graph.get_graph().draw_mermaid_png(
        output_file_path=output_file_path,
    )
    typer.echo(f"Graph saved to {output_file_path}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
