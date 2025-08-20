import logging

import typer
from dotenv import load_dotenv

from template_langgraph.agents.demo_agents.multi_agent import graph as multi_agent_graph
from template_langgraph.agents.demo_agents.parallel_rag_agent.agent import graph as parallel_rag_agent_graph
from template_langgraph.agents.demo_agents.weather_agent import graph as weather_agent_graph
from template_langgraph.loggers import get_logger

app = typer.Typer(
    add_completion=False,
    help="Demo Agents CLI",
)
logger = get_logger(__name__)


@app.command()
def weather_agent(
    query: str = typer.Option(
        "What's the weather in Japan?",
        "--query",
        "-q",
        help="The query to ask the model",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    response = weather_agent_graph.invoke(
        {
            "messages": [
                {"role": "user", "content": query},
            ],
        },
        debug=True,
    )
    logger.info(response["messages"][-1].content)


@app.command()
def multi_agent(
    query: str = typer.Option(
        "What's the weather in Japan?",
        "--query",
        "-q",
        help="The query to ask the model",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    response = multi_agent_graph.invoke(
        {
            "messages": [
                {"role": "user", "content": query},
            ],
        },
        debug=True,
    )
    logger.info(response["messages"][-1].content)


@app.command()
def parallel_rag_agent(
    query: str = typer.Option(
        "KABUTO のシステム概要やトラブルシュート事例を多種多様な情報ソースから回答して",
        "--query",
        "-q",
        help="The query to decompose into tasks",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    for event in parallel_rag_agent_graph.stream(
        input={
            "query": query,
        },
        debug=True,
    ):
        logger.info("-" * 20)
        logger.info(f"Event: {event}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
