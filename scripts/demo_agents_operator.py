import logging

import typer
from dotenv import load_dotenv

from template_langgraph.agents.demo_agents.multi_agent import app as multi_agent_app
from template_langgraph.agents.demo_agents.parallel_processor_agent import app as parallel_agent_app
from template_langgraph.agents.demo_agents.weather_agent import app as weather_agent_app
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

    response = weather_agent_app.invoke(
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

    response = multi_agent_app.invoke(
        {
            "messages": [
                {"role": "user", "content": query},
            ],
        },
        debug=True,
    )
    logger.info(response["messages"][-1].content)


@app.command()
def parallel_agent(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    parallel_agent_app.invoke(
        {
            "messages": [
                {"role": "user", "content": "Simulate multiple tasks in parallel"},
            ],
            "tasks": [
                "Task 1",
                "Task 2",
                "Task 3",
            ],
        },
        debug=True,
    )


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
