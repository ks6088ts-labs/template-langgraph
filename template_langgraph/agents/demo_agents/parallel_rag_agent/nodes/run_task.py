import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools.base import BaseTool

from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    Task,
    TaskResult,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class RunTask:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
    ):
        self.llm = llm
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: dict) -> dict:
        logger.info(f"Running state... {state}")
        task: Task = state.get("task", None)
        query: str = state.get("query", None)
        logger.info(f"Task: {task.model_dump_json(indent=2)}")

        try:
            observation = self.tools_by_name[task.tool_name].invoke(task.tool_args)
        except Exception as e:
            logger.error(f"Error occurred while invoking tools: {e}")
            observation = {"error": str(e)}

        result = self.llm.invoke(
            input=[
                HumanMessage(content=query),
                HumanMessage(
                    content=json.dumps(observation.__str__(), ensure_ascii=False),
                ),
            ],
        )

        logger.info(f"LLM response: {result.model_dump_json(indent=2)}, type: {type(result)}")

        result = TaskResult(
            task=task,
            result_code=0,
            message=result.content,
        )

        return {
            "task_results": [result],
        }
