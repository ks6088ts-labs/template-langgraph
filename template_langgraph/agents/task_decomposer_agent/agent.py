from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from template_langgraph.agents.task_decomposer_agent.models import AgentState, TaskList
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
        workflow.add_node("human_feedback", self.human_feedback)

        # Create edges
        workflow.set_entry_point("chat")
        workflow.add_edge("chat", "human_feedback")
        workflow.add_conditional_edges(
            source="human_feedback",
            path=self.route_human_feedback,
            path_map={
                "loopback": "chat",
                "end": END,
            },
        )
        return workflow.compile()

    def chat(self, state: AgentState) -> AgentState:
        """Chat with tools using the state."""
        logger.info(f"Chatting with tools using state: {state}")

        task_list = self.llm.with_structured_output(TaskList).invoke(
            input=state["messages"],
        )
        state["task_list"] = task_list
        logger.info(f"Decomposed tasks: {task_list}")
        return state

    def human_feedback(self, state: AgentState) -> AgentState:
        """Handle human feedback."""
        logger.info(f"Handling human feedback with state: {state}")
        feedback = interrupt("Type your feedback. If you want to end the conversation, type 'end'.")
        state["messages"].append(
            {
                "content": feedback,
                "role": "user",
            }
        )
        return state

    def route_human_feedback(
        self,
        state: AgentState,
    ):
        """
        Use in the conditional_edge to route to the HumanFeedbackNode if the last message
        has human feedback. Otherwise, route to the end.
        """
        human_feedback = state["messages"][-1].content.strip().lower()
        if human_feedback == "end":
            logger.info("Ending the conversation as per user request.")
            return "end"
        logger.info("Looping back to chat for further processing.")
        return "loopback"

    def draw_mermaid_png(self) -> bytes:
        """Draw the graph in Mermaid format."""
        return self.create_graph().get_graph().draw_mermaid_png()


graph = TaskDecomposerAgent().create_graph()
