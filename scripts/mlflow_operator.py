import logging
import os
from logging import basicConfig

import mlflow
import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

from template_langgraph.agents.demo_agents.weather_agent import graph
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper, Settings
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
    experiment_name: str = typer.Option(
        "LangGraph Experiment",
        "--experiment-name",
        "-e",
        help="MLflow experiment name",
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
    mlflow.set_experiment(experiment_name)

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


@app.command(
    help="Evaluate the LangGraph agent with MLflow tracing ref. https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/"
)
def evaluate(
    experiment_name: str = typer.Option(
        "LangGraph Experiment",
        "--experiment-name",
        "-e",
        help="MLflow experiment name",
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
    mlflow.set_experiment(experiment_name)

    llm = AzureOpenAiWrapper().chat_model

    def qa_predict_fn(question: str) -> str:
        """Simple Q&A prediction function using OpenAI"""
        response = llm.invoke(
            [
                SystemMessage(content="You are a helpful assistant. Answer questions concisely."),
                HumanMessage(content=question),
            ]
        )
        return response.content

    @scorer
    def is_concise(outputs: str) -> bool:
        """Evaluate if the answer is concise (less than 5 words)"""
        return len(outputs.split()) <= 5

    # To configure LiteLLM for Azure OpenAI ref. https://docs.litellm.ai/docs/providers/azure/
    settings = Settings()

    os.environ["AZURE_API_KEY"] = settings.azure_openai_api_key
    os.environ["AZURE_API_BASE"] = settings.azure_openai_endpoint
    os.environ["AZURE_API_VERSION"] = settings.azure_openai_api_version
    os.environ["AZURE_API_TYPE"] = "azure"

    model = f"azure:/{settings.azure_openai_model_chat}"
    results = mlflow.genai.evaluate(
        data=[
            {
                "inputs": {"question": "What is the capital of France?"},
                "expectations": {"expected_response": "Paris"},
            },
            {
                "inputs": {"question": "Who was the first person to build an airplane?"},
                "expectations": {"expected_response": "Wright Brothers"},
            },
            {
                "inputs": {"question": "Who wrote Romeo and Juliet?"},
                "expectations": {"expected_response": "William Shakespeare"},
            },
        ],
        predict_fn=qa_predict_fn,
        scorers=[
            Correctness(model=model),
            Guidelines(
                model=model,
                name="is_english",
                guidelines="The answer must be in English",
            ),
            is_concise,
        ],
    )
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
