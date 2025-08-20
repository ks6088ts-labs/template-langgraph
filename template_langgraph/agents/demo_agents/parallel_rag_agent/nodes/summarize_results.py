from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    ParallelRagAgentState,
    TaskResult,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class SummarizeResults:
    def __init__(self):
        pass

    def __call__(self, state: ParallelRagAgentState) -> dict:
        logger.info(f"Summarizing results... {state}")
        task_results: list[TaskResult] = state.get("task_results", [])
        summary = ""
        for task_result in task_results:
            summary += f"Tool: {task_result['task'].tool_name}: {task_result['message']}\n------\n"
        logger.info(f"Final summary: {summary}")
        return {
            "summary": summary,
        }
