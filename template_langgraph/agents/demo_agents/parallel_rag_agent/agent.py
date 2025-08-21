from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools.base import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    ParallelRagAgentInputState,
    ParallelRagAgentOutputState,
    ParallelRagAgentState,
)
from template_langgraph.agents.demo_agents.parallel_rag_agent.nodes.decompose_tasks import DecomposeTasks
from template_langgraph.agents.demo_agents.parallel_rag_agent.nodes.run_task import RunTask
from template_langgraph.agents.demo_agents.parallel_rag_agent.nodes.summarize_results import SummarizeResults
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.tools.common import get_default_tools


class ParallelRagAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        system_prompts_by_name: dict[str, str],
    ):
        self.llm = llm
        self.decompose_tasks = DecomposeTasks(
            llm=llm,
            tools=tools,
            system_prompts_by_name=system_prompts_by_name,
        )
        self.run_task = RunTask(
            llm=llm,
            tools=tools,
        )
        self.summarize_results = SummarizeResults()

    def create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            state_schema=ParallelRagAgentState,
            input_schema=ParallelRagAgentInputState,
            output_schema=ParallelRagAgentOutputState,
        )
        workflow.add_node("decompose_tasks", self.decompose_tasks)
        workflow.add_node("run_task", self.run_task)
        workflow.add_node("summarize_results", self.summarize_results)

        workflow.add_edge("run_task", "summarize_results")
        workflow.set_entry_point("decompose_tasks")
        workflow.set_finish_point("summarize_results")
        return workflow.compile()


# FIXME: Update system prompts for each tool
SYSTEM_PROMPT_BY_NAME = {tool.name: f"THIS IS DEFAULT SYSTEM PROMPT FOR {tool.name}" for tool in get_default_tools()}

graph = ParallelRagAgent(
    llm=AzureOpenAiWrapper().chat_model,
    tools=get_default_tools(),
    system_prompts_by_name=SYSTEM_PROMPT_BY_NAME,
).create_graph()
