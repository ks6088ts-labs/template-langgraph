import logging

from template_langgraph.agents.chat_with_tools_agent.agent import AgentState
from template_langgraph.agents.chat_with_tools_agent.agent import graph as chat_with_tools_agent_graph
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def stream_graph_updates(
    state: AgentState,
) -> dict:
    for event in chat_with_tools_agent_graph.stream(input=state):
        logger.info("-" * 20)
        logger.info(f"Event: {event}")
    return event


if __name__ == "__main__":
    user_input = input("User: ")
    for event in chat_with_tools_agent_graph.stream(
        input=AgentState(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
        )
    ):
        logger.info("-" * 20)
        logger.info(f"Event: {event}")
