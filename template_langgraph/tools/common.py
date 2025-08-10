from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.ai_search_tool import search_ai_search
from template_langgraph.tools.cosmosdb_tool import search_cosmosdb
from template_langgraph.tools.dify_tool import run_dify_workflow
from template_langgraph.tools.elasticsearch_tool import search_elasticsearch
from template_langgraph.tools.qdrant_tool import search_qdrant
from template_langgraph.tools.sql_database_tool import SqlDatabaseClientWrapper

logger = get_logger(__name__)


def get_default_tools():
    try:
        sql_database_tools = SqlDatabaseClientWrapper().get_tools(
            llm=AzureOpenAiWrapper().chat_model,
        )
    except Exception as e:
        logger.error(f"Error occurred while getting SQL database tools: {e}")
        sql_database_tools = []
    return [
        search_ai_search,
        search_cosmosdb,
        run_dify_workflow,
        search_qdrant,
        search_elasticsearch,
    ] + sql_database_tools
