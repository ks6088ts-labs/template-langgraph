import asyncio
import json

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from template_langgraph.agents.chat_with_tools_agent.models import AgentState
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.common import get_default_tools, is_async_call_required

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
            try:
                if is_async_call_required(tool_call["name"]):
                    observation = asyncio.run(self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"]))
                else:
                    observation = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(observation.__str__(), ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                logger.error(f"Error occurred while invoking tools: {e}")
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}, ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": outputs}


class ChatWithToolsAgent:
    def __init__(
        self,
        tools=get_default_tools(),
        checkpointer=None,
        store=None,
        system_prompt: str | None = None,
    ):
        self.llm = AzureOpenAiWrapper().chat_model
        self.tools = tools
        self.checkpointer = checkpointer
        self.store = store
        self.system_prompt = system_prompt
        self._system_message = SystemMessage(content=system_prompt) if system_prompt else None

    def create_graph(self):
        """Create the main graph for the agent."""
        # Create the workflow state graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("chat_with_tools", self.chat_with_tools)
        workflow.add_node(
            "tools",
            BasicToolNode(
                tools=self.tools,
            ),
        )

        # Create edges
        workflow.set_entry_point("chat_with_tools")
        workflow.add_conditional_edges(
            source="chat_with_tools",
            path=self.route_tools,
            path_map={
                "tools": "tools",
                END: END,
            },
        )
        workflow.add_edge("tools", "chat_with_tools")

        # Compile the graph
        return workflow.compile(
            name=ChatWithToolsAgent.__name__,
            checkpointer=self.checkpointer,
            store=self.store,
        )

    def chat_with_tools(self, state: AgentState) -> AgentState:
        """Chat with tools using the state."""
        logger.info(f"Chatting with tools using state: {state}")
        llm_with_tools = self.llm.bind_tools(
            tools=self.tools,
        )
        messages = self._prepare_messages(state)
        return {
            "messages": [
                llm_with_tools.invoke(messages),
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

    def _prepare_messages(self, state: AgentState):
        """Return a message list with the optional system prompt prefixed."""
        base_messages = list(state) if isinstance(state, list) else list(state.get("messages", []))
        if not self._system_message:
            return base_messages
        if base_messages and isinstance(base_messages[0], SystemMessage):
            if base_messages[0].content == self._system_message.content:
                return base_messages
        return [self._system_message, *base_messages]


graph = ChatWithToolsAgent().create_graph()
