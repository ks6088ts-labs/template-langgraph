from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.ai_search_tool import search_ai_search
from template_langgraph.tools.cosmosdb_tool import search_cosmosdb
from template_langgraph.tools.dify_tool import run_dify_workflow
from template_langgraph.tools.elasticsearch_tool import search_elasticsearch
from template_langgraph.tools.mcp_tool import McpClientWrapper
from template_langgraph.tools.qdrant_tool import search_qdrant
from template_langgraph.tools.sql_database_tool import SqlDatabaseClientWrapper

logger = get_logger(__name__)
mcp_tools = McpClientWrapper().get_tools()


def get_default_tools():
    return (
        [
            search_ai_search,
            search_cosmosdb,
            run_dify_workflow,
            search_qdrant,
            search_elasticsearch,
        ]
        + SqlDatabaseClientWrapper().get_tools(
            llm=AzureOpenAiWrapper().chat_model,
        )
        + mcp_tools
    )


def is_async_call_required(tool_name: str) -> bool:
    mcp_tool_names = [tool.name for tool in mcp_tools]
    return tool_name in [
        *mcp_tool_names,
    ]
