from langgraph.graph import END, StateGraph

from template_langgraph.agents.chat_with_tools_agent.models import AgentState
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class TaskDecomposerAgent:
    def __init__(self):
        self.llm = AzureOpenAiWrapper().chat_model

    def create_graph(self):
        """Create the main graph for the agent."""
        # Create the workflow state graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("chat", self.chat)

        # Create edges
        workflow.set_entry_point("chat")
        workflow.add_edge("chat", END)

        # Compile the graph
        return workflow.compile()

    def chat(self, state: AgentState) -> AgentState:
        """Chat with tools using the state."""
        logger.info(f"Chatting with tools using state: {state}")
        return {
            "messages": [
                self.llm.invoke(state["messages"]),
            ]
        }

    def draw_mermaid_png(self) -> bytes:
        """Draw the graph in Mermaid format."""
        return self.create_graph().get_graph().draw_mermaid_png()


graph = TaskDecomposerAgent().create_graph()
