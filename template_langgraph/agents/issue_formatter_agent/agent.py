from langgraph.graph import StateGraph

from template_langgraph.agents.issue_formatter_agent.models import AgentState, Issue
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class IssueFormatterAgent:
    def __init__(self):
        self.llm = AzureOpenAiWrapper().chat_model

    def create_graph(self):
        """Create the main graph for the agent."""
        # Create the workflow state graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("analyze", self.analyze)

        # Create edges
        workflow.set_entry_point("analyze")
        workflow.set_finish_point("analyze")

        # Compile the graph
        return workflow.compile(
            name=IssueFormatterAgent.__name__,
        )

    def analyze(self, state: AgentState) -> AgentState:
        """Analyze the issue and extract relevant information."""
        logger.info(f"Analyzing issue with state: {state}")
        issue = self.llm.with_structured_output(Issue).invoke(
            input=state["messages"],
        )
        state["issue"] = issue
        return state


graph = IssueFormatterAgent().create_graph()
