from langgraph_supervisor import create_supervisor

from template_langgraph.agents.chat_with_tools_agent.agent import graph as chat_with_tools_agent_graph
from template_langgraph.agents.issue_formatter_agent.agent import graph as issue_formatter_agent_graph
from template_langgraph.agents.task_decomposer_agent.agent import graph as task_decomposer_agent_graph
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)

PROMPT = """
KABUTO に関する質問に答えるために、必要な情報を収集し適切な回答を提供します。
- 過去の FAQ やドキュメントを参照して、質問に対する答えを見つける必要がある場合は Chat with Tools Agent を使用します。
- 質問の内容を整理し、適切な形式で回答するために Issue Formatter Agent を使用します。
- 質問が複雑であったり追加の情報が必要な場合は、 Task Decomposer Agent を使用してタスクを分解し、順次処理します。
- 質問が明確でない場合は、追加の情報を求めるための質問を行います。
- 質問が単純な場合は、直接回答を提供します。
"""


class SupervisorAgent:
    def __init__(self):
        self.agent = create_supervisor(
            agents=[
                chat_with_tools_agent_graph,
                issue_formatter_agent_graph,
                task_decomposer_agent_graph,
            ],
            model=AzureOpenAiWrapper().chat_model,
            prompt=PROMPT,
            debug=True,
            supervisor_name=SupervisorAgent.__name__,
        )


graph = SupervisorAgent().agent.compile()
