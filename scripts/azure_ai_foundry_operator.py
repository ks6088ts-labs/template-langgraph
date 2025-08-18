import logging

import typer
from dotenv import load_dotenv

from template_langgraph.llms.azure_ai_foundrys import AzureAiFoundryWrapper
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Azure AI Foundry operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def chat(
    query: str = typer.Option(
        "Hello",
        "--query",
        "-q",
        help="The query to send to the AI",
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

    logger.info("Running Azure AI Foundry chat...")

    wrapper = AzureAiFoundryWrapper()

    openai_client = wrapper.get_openai_client()

    response = openai_client.chat.completions.create(
        model=wrapper.settings.azure_ai_foundry_inference_model_chat,
        messages=[
            {"role": "user", "content": query},
        ],
    )
    logger.info(response.choices[0].message.content)


@app.command()
def create_agent(
    name: str = typer.Option(
        "MyAgent",
        "--name",
        "-n",
        help="The name of the agent",
    ),
    instructions: str = typer.Option(
        "This is my agent",
        "--instructions",
        "-i",
        help="The instructions for the agent",
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

    logger.info("Creating agent...")
    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    with project_client.agents as agents_client:
        # Create a new agent
        agent = agents_client.create_agent(
            name=name,
            instructions=instructions,
            model=AzureAiFoundryWrapper().settings.azure_ai_foundry_inference_model_chat,
        )
        logger.info(f"Created agent: {agent.as_dict()}")


@app.command()
def delete_agent(
    agent_id: str = typer.Option(
        "asst_xxx",
        "--agent-id",
        "-a",
        help="The ID of the agent to delete",
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

    logger.info("Deleting agent...")
    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    with project_client.agents as agents_client:
        agents_client.delete_agent(agent_id=agent_id)


@app.command()
def list_agents(
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

    logger.info("Listing agents...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    with project_client.agents as agents_client:
        agents = agents_client.list_agents()
        for agent in agents:
            logger.info(f"Agent ID: {agent.id}, Name: {agent.name}")


@app.command()
def create_thread(
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

    logger.info("Creating thread...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    thread = project_client.agents.threads.create()
    logger.info(thread.as_dict())


@app.command()
def delete_thread(
    thread_id: str = typer.Option(
        "thread_xxx",
        "--thread-id",
        "-t",
        help="The ID of the thread to delete",
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

    logger.info("Deleting thread...")
    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    project_client.agents.threads.delete(thread_id=thread_id)


@app.command()
def list_threads(
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

    logger.info("Listing threads...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    threads = project_client.agents.threads.list()
    for thread in threads:
        logger.info(thread.as_dict())


@app.command()
def create_message(
    role: str = typer.Option(
        "user",
        "--role",
        "-r",
        help="The role of the message sender",
    ),
    content: str = typer.Option(
        "Hello, world!",
        "--content",
        "-c",
        help="The content of the message",
    ),
    thread_id: str = typer.Option(
        "thread_xxx",
        "--thread-id",
        "-t",
        help="The ID of the thread to list messages from",
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

    logger.info("Creating message...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    message = project_client.agents.messages.create(
        thread_id=thread_id,
        role=role,
        content=content,
    )
    logger.info(message.as_dict())


@app.command()
def list_messages(
    thread_id: str = typer.Option(
        "thread_xxx",
        "--thread-id",
        "-t",
        help="The ID of the thread to list messages from",
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

    logger.info("Listing messages...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    messages = project_client.agents.messages.list(thread_id=thread_id)
    for message in messages:
        logger.info(message.as_dict())


@app.command()
def run_thread(
    agent_id: str = typer.Option(
        "agent_xxx",
        "--agent-id",
        "-a",
        help="The ID of the agent to run",
    ),
    thread_id: str = typer.Option(
        "thread_xxx",
        "--thread-id",
        "-t",
        help="The ID of the thread to run",
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

    logger.info("Running thread...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    run = project_client.agents.runs.create(
        thread_id=thread_id,
        agent_id=agent_id,
    )

    logger.info(f"Run created: {run.as_dict()}")


@app.command()
def get_run(
    thread_id: str = typer.Option(
        "thread_xxx",
        "--thread-id",
        "-t",
        help="The ID of the thread to run",
    ),
    run_id: str = typer.Option(
        "run_xxx",
        "--run-id",
        "-r",
        help="The ID of the run to retrieve",
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

    logger.info("Getting run...")

    project_client = AzureAiFoundryWrapper().get_ai_project_client()
    run = project_client.agents.runs.get(
        thread_id=thread_id,
        run_id=run_id,
    )
    logger.info(run.as_dict())


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
