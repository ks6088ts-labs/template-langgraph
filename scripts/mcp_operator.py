import asyncio
import logging
from logging import basicConfig

import typer
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="MCP operator CLI",
)

# Set up logging
logger = get_logger(__name__)


def set_verbose_logging(verbose: bool):
    if verbose:
        logger.setLevel(logging.DEBUG)
        basicConfig(level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)


async def run_agent(query: str) -> dict:
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["./template_langgraph/mcps/math_server.py"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    llm = AzureOpenAiWrapper().chat_model

    def call_model(state: MessagesState):
        response = llm.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return await graph.ainvoke({"messages": query})


@app.command()
def chat(
    query: str = typer.Option(
        "What's (3 + 5) x 12?",
        "--query",
        "-q",
        help="Input query to the chatbot",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    set_verbose_logging(verbose)
    logger.info("Running...")
    logger.info(f"Query: {query}")
    response = asyncio.run(run_agent(query=query))
    for k, v in response.items():
        logger.info(f"{k}: {v}")
    logger.info(response.get("messages")[-1].content)


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
