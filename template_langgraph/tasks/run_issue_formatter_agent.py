import logging

from template_langgraph.agents.issue_formatter_agent.agent import AgentState, graph
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def stream_graph_updates(
    state: AgentState,
) -> dict:
    for event in graph.stream(input=state):
        logger.info("-" * 20)
        logger.info(f"Event: {event}")
    return event


if __name__ == "__main__":
    user_input = input("User: ")
    state = AgentState(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        profile=None,
    )
    last_event = stream_graph_updates(state)
    for value in last_event.values():
        logger.info(f"Formatted issue: {value['issue']}")
