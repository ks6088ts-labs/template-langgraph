import logging
from logging import basicConfig

import mlflow
import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from template_langgraph.agents.demo_agents.weather_agent import graph
from template_langgraph.loggers import get_logger

app = typer.Typer(
    add_completion=False,
    help="MLflow operator CLI",
)
logger = get_logger(__name__)


def set_verbose_logging(verbose: bool):
    if verbose:
        logger.setLevel(logging.DEBUG)
        basicConfig(level=logging.DEBUG)


@app.command(
    help="Run the LangGraph agent with MLflow tracing ref. https://mlflow.org/docs/2.21.3/tracing/integrations/langgraph"
)
def tracing(
    query: str = typer.Option(
        "What is the weather like in Japan?",
        "--query",
        "-q",
        help="Query to run with the LangGraph agent",
    ),
    tracking_uri: str = typer.Option(
        "http://localhost:5001",
        "--tracking-uri",
        "-t",
        help="MLflow tracking URI",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    set_verbose_logging(verbose)
    logger.info("Running...")

    mlflow.langchain.autolog()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("LangGraph Experiment")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=query),
            ]
        },
    )
    logger.info(f"Result: {result}")

    # Get the trace object just created
    trace = mlflow.get_trace(
        trace_id=mlflow.get_last_active_trace_id(),
    )
    logger.info(f"Trace info: {trace.info.token_usage}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
