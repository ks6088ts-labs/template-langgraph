from langgraph.graph import END, StateGraph

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
        workflow.add_edge("analyze", END)

        # Compile the graph
        return workflow.compile()

    def analyze(self, state: AgentState) -> AgentState:
        """Analyze the issue and extract relevant information."""
        logger.info(f"Analyzing issue with state: {state}")
        issue = self.llm.with_structured_output(Issue).invoke(
            input=state["messages"],
        )
        state["issue"] = issue
        return state

    def draw_mermaid_png(self) -> bytes:
        """Draw the graph in Mermaid format."""
        return self.create_graph().get_graph().draw_mermaid_png()


graph = IssueFormatterAgent().create_graph()
