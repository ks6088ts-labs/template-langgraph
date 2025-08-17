from template_langgraph.agents.demo_agents.parallel_processor_agent.models import (
    ParallelProcessorAgentState,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class SummarizeResults:
    def __init__(self):
        pass

    def __call__(self, state: ParallelProcessorAgentState) -> dict:
        logger.info(f"Summarizing results... {state}")
        task_results = state.get("task_results", "")
        logger.info(f"Task results: {task_results}")
        return {
            "summary": task_results.__str__(),
        }
