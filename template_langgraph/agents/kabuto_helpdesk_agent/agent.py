from langgraph.prebuilt import create_react_agent

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.common import DEFAULT_TOOLS

logger = get_logger(__name__)


class KabutoHelpdeskAgent:
    def __init__(self, tools=DEFAULT_TOOLS):
        self.agent = create_react_agent(
            model=AzureOpenAiWrapper().chat_model,
            tools=tools,
            prompt="KABUTO に関する質問に答えるために、必要な情報を収集し適切な回答を提供します",
            debug=True,
        )


graph = KabutoHelpdeskAgent().agent
