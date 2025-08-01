from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from template_langgraph.agents.basic_workflow_agent.models import AgentInput, AgentOutput, AgentState, Profile
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class BasicWorkflowAgent:
    def __init__(self):
        self.llm = AzureOpenAiWrapper().chat_model

    def create_graph(self):
        """Create the main graph for the agent."""
        # Create the workflow state graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("do_something", self.do_something)
        workflow.add_node("extract_profile", self.extract_profile)
        workflow.add_node("finalize", self.finalize)

        # Create edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "do_something")
        workflow.add_edge("do_something", "extract_profile")
        workflow.add_edge("extract_profile", "finalize")
        workflow.add_edge("finalize", END)

        # Compile the graph
        return workflow.compile()

    def initialize(self, state: AgentState) -> AgentState:
        """Initialize the agent with the given state."""
        logger.info(f"Initializing BasicWorkflowAgent with state: {state}")
        # Here you can add any initialization logic if needed
        return state

    def do_something(self, state: AgentState) -> AgentState:
        """Perform some action with the given state."""
        logger.info(f"Doing something with state: {state}")

        # Here you can add the logic for the action
        response: AIMessage = self.llm.invoke(
            input=state["messages"],
        )
        logger.info(f"Response after doing something: {response}")
        state["messages"].append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        return state

    def extract_profile(self, state: AgentState) -> AgentState:
        """Extract profile information from the state."""
        logger.info(f"Extracting profile from state: {state}")
        profile = self.llm.with_structured_output(Profile).invoke(
            input=state["messages"],
        )
        state["profile"] = profile
        return state

    def finalize(self, state: AgentState) -> AgentState:
        """Finalize the agent's work and prepare the output."""
        logger.info(f"Finalizing BasicWorkflowAgent with state: {state}")
        # Here you can add any finalization logic if needed
        return state

    def run_agent(self, input: AgentInput) -> AgentOutput:
        """Run the agent with the given input."""
        logger.info(f"Running BasicWorkflowAgent with question: {input.model_dump_json(indent=2)}")
        app = self.create_graph()
        initial_state: AgentState = {
            "messages": [
                {
                    "role": "user",
                    "content": input.request,
                }
            ],
        }
        final_state = app.invoke(initial_state)
        logger.info(f"Final state after running agent: {final_state}")
        return AgentOutput(
            response=final_state["messages"][-1].content,
            profile=final_state["profile"],
        )

    def draw_mermaid_png(self) -> bytes:
        """Draw the graph in Mermaid format."""
        return self.create_graph().get_graph().draw_mermaid_png()


graph = BasicWorkflowAgent().create_graph()
