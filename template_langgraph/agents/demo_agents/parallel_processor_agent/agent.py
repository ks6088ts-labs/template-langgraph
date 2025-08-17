from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from template_langgraph.agents.demo_agents.parallel_processor_agent.models import (
    ParallelProcessorAgentInputState,
    ParallelProcessorAgentOutputState,
    ParallelProcessorAgentState,
)
from template_langgraph.agents.demo_agents.parallel_processor_agent.nodes.decompose_tasks import DecomposeTasks
from template_langgraph.agents.demo_agents.parallel_processor_agent.nodes.run_task import RunTask
from template_langgraph.agents.demo_agents.parallel_processor_agent.nodes.summarize_results import SummarizeResults
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


class ParallelProcessorAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.decompose_tasks = DecomposeTasks(llm)
        self.run_task = RunTask()
        self.summarize_results = SummarizeResults()

    def create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            state_schema=ParallelProcessorAgentState,
            input_schema=ParallelProcessorAgentInputState,
            output_schema=ParallelProcessorAgentOutputState,
        )
        workflow.add_node("decompose_tasks", self.decompose_tasks)
        workflow.add_node("run_task", self.run_task)
        workflow.add_node("summarize_results", self.summarize_results)

        workflow.add_edge("run_task", "summarize_results")
        workflow.set_entry_point("decompose_tasks")
        workflow.set_finish_point("summarize_results")
        return workflow.compile()


app = ParallelProcessorAgent(AzureOpenAiWrapper().chat_model).create_graph()
