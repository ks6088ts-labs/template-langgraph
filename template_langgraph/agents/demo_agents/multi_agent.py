from typing import Literal

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from template_langgraph.agents.demo_agents.weather_agent import app
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


def transfer_to_weather_agent():
    """Call this to transfer to the weather agent"""


tools = [transfer_to_weather_agent]
llm = AzureOpenAiWrapper().chat_model.bind_tools(tools=tools)


def call_model(state: MessagesState) -> Command[Literal["weather_agent", END]]:
    messages = state["messages"]
    response = llm.invoke(messages)
    if len(response.tool_calls) > 0:
        return Command(
            goto="weather_agent",
        )
    else:
        return Command(
            goto=END,
            update={
                "messages": [response],
            },
        )


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("weather_agent", app)
workflow.add_edge(START, "agent")
app = workflow.compile()
