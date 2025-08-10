from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.tools.cosmosdb_tool import search_cosmosdb
from template_langgraph.tools.dify_tool import run_dify_workflow
from template_langgraph.tools.elasticsearch_tool import search_elasticsearch
from template_langgraph.tools.qdrant_tool import search_qdrant
from template_langgraph.tools.sql_database_tool import SqlDatabaseClientWrapper

DEFAULT_TOOLS = [
    search_cosmosdb,
    run_dify_workflow,
    search_qdrant,
    search_elasticsearch,
] + SqlDatabaseClientWrapper().get_tools(
    llm=AzureOpenAiWrapper().chat_model,
)
