import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from template_langgraph.agents.basic_workflow_agent.models import AgentInput, AgentOutput, AgentState, Profile
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.elasticsearch_tool import search_elasticsearch
from template_langgraph.tools.qdrants import search_qdrant

logger = get_logger(__name__)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result.__str__(), ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


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
        workflow.add_node("chat_with_tools", self.chat_with_tools)
        workflow.add_node(
            "tools",
            BasicToolNode(
                tools=[
                    search_qdrant,
                    search_elasticsearch,
                ]
            ),
        )
        workflow.add_node("finalize", self.finalize)

        # Create edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "do_something")
        workflow.add_edge("do_something", "extract_profile")
        workflow.add_edge("extract_profile", "chat_with_tools")
        workflow.add_conditional_edges(
            "chat_with_tools",
            self.route_tools,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            # e.g., "tools": "my_tools"
            {"tools": "tools", END: "finalize"},
        )
        workflow.add_edge("tools", "chat_with_tools")
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

    def chat_with_tools(self, state: AgentState) -> AgentState:
        """Chat with tools using the state."""
        logger.info(f"Chatting with tools using state: {state}")
        llm_with_tools = self.llm.bind_tools(
            tools=[
                search_qdrant,
                search_elasticsearch,
            ],
        )
        return {
            "messages": [
                llm_with_tools.invoke(state["messages"]),
            ]
        }

    def route_tools(
        self,
        state: AgentState,
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

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
