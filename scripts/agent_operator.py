import logging
from uuid import uuid4

import typer
from dotenv import load_dotenv

from template_langgraph.agents.chat_with_tools_agent.agent import graph as chat_with_tools_agent_graph
from template_langgraph.agents.issue_formatter_agent.agent import graph as issue_formatter_agent_graph
from template_langgraph.agents.kabuto_helpdesk_agent.agent import graph as kabuto_helpdesk_agent_graph
from template_langgraph.agents.news_summarizer_agent.agent import (
    graph as news_summarizer_agent_graph,
)
from template_langgraph.agents.news_summarizer_agent.models import Article
from template_langgraph.agents.task_decomposer_agent.agent import graph as task_decomposer_agent_graph
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="agent runner CLI",
)

# Set up logging
logger = get_logger(__name__)


def get_agent_graph(name: str):
    if name == "chat_with_tools_agent":
        return chat_with_tools_agent_graph
    elif name == "issue_formatter_agent":
        return issue_formatter_agent_graph
    elif name == "task_decomposer_agent":
        return task_decomposer_agent_graph
    elif name == "kabuto_helpdesk_agent":
        return kabuto_helpdesk_agent_graph
    elif name == "news_summarizer_agent":
        return news_summarizer_agent_graph
    else:
        raise ValueError(f"Unknown agent name: {name}")


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
    typer.echo(f"Drawing agent: {name}")
    get_agent_graph(name).get_graph().draw_mermaid_png(
        output_file_path=output_file_path,
    )
    typer.echo(f"Graph saved to {output_file_path}")


@app.command()
def run(
    name: str = typer.Option(
        "chat_with_tools_agent",
        "--name",
        "-n",
        help="Name of the agent to draw",
    ),
    question: str = typer.Option(
        "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。",
        "--question",
        "-q",
        help="Question to ask the agent",
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

    assert name not in [
        "news_summarizer_agent",
    ], f"{name} is not supported. Please use another agent."

    graph = get_agent_graph(name)
    for event in graph.stream(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
        }
    ):
        logger.info("-" * 20)
        logger.info(f"Event: {event}")


@app.command()
def news_summarizer_agent(
    prompt: str = typer.Option(
        "Please summarize the articles in Japanese briefly in 3 sentences.",
        "--prompt",
        "-p",
        help="Prompt for the agent",
    ),
    urls: str = typer.Option(
        "https://example.com/article1,https://example.com/article2",
        "--urls",
        "-u",
        help="Comma-separated list of URLs to summarize",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    from template_langgraph.agents.news_summarizer_agent.models import (
        AgentInputState,
        AgentState,
    )

    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    graph = news_summarizer_agent_graph
    for event in graph.stream(
        input=AgentState(
            input=AgentInputState(
                prompt=prompt,
                id=str(uuid4()),
                urls=urls.split(",") if urls else [],
            ),
            articles=[],
        )
    ):
        logger.info("-" * 20)
        logger.info(f"Event: {event}")

    articles: list[Article] = event["notify"]["articles"]
    for article in articles:
        logger.info(f"{article.structured_article.model_dump_json(indent=2)}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
