from langgraph.prebuilt import create_react_agent

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.elasticsearch_tool import search_elasticsearch
from template_langgraph.tools.qdrants import search_qdrant

logger = get_logger(__name__)


class KabutoHelpdeskAgent:
    def __init__(self, tools=None):
        if tools is None:
            # Default tool for searching Qdrant
            tools = [
                search_qdrant,
                search_elasticsearch,
                # Add other tools as needed
            ]
        self.agent = create_react_agent(
            model=AzureOpenAiWrapper().chat_model,
            tools=tools,
            prompt="KABUTO に関する質問に答えるために、必要な情報を収集し適切な回答を提供します",
            debug=True,
        )

    def run(self, question: str) -> dict:
        logger.info(f"Running KabutoHelpdeskAgent with question: {question}")
        return self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ]
            }
        )


graph = KabutoHelpdeskAgent().agent
