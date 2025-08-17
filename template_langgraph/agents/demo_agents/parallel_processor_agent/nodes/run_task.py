from template_langgraph.agents.demo_agents.parallel_processor_agent.models import (
    Task,
    TaskResult,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class RunTask:
    def __init__(self):
        pass

    def __call__(self, state: dict) -> dict:
        logger.info(f"Running state... {state}")
        task: Task = state.get("task", None)
        logger.info(f"Task: {task.model_dump_json(indent=2)}")

        # FIXME: Simulate task processing for now. Replace with actual processing logic e.g. Tool call agent
        import time

        time.sleep(3)
        result = TaskResult(
            task=task,
            message="Task completed successfully",
            result_code=0,
        )

        return {
            "task_results": [result],
        }
