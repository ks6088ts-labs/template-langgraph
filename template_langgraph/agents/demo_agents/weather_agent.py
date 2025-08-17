from typing import Literal

from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


@tool
def search(query: str) -> str:
    """Call to surf the web"""
    if "japan" in query.lower():
        return "It's 60 degrees and cloudy in Japan"
    return "It's 90 degrees and sunny in Japan"


tools = [search]
tool_node = ToolNode(tools=tools)
llm = AzureOpenAiWrapper().chat_model.bind_tools(tools=tools)


def call_model(state: MessagesState) -> Command[Literal["tools", END]]:
    messages = state["messages"]
    response = llm.invoke(messages)
    if len(response.tool_calls) > 0:
        next_node = "tools"
    else:
        next_node = END
    return Command(
        goto=next_node,
        update={
            "messages": [response],
        },
    )


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
app = workflow.compile()
